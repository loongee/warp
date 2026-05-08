//! Manages the embedded Python proxy server for local DeepSeek AI conversations.
//!
//! On startup:
//! 1. Extracts embedded proxy Python source to <data_dir>/proxy/
//! 2. Creates a Python venv and installs dependencies (first run only)
//! 3. Finds a free port dynamically
//! 4. Spawns uvicorn as a child process on that port
//!
//! On shutdown:
//! 1. Sends POST /shutdown to the proxy (graceful exit)
//! 2. Waits up to 3 seconds for the child process to exit
//! 3. Falls back to SIGKILL if it doesn't exit in time

use std::fs;
use std::io::Read as _;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicI32, AtomicU16, Ordering};
use std::time::{Duration, Instant};

use crate::Assets;

/// SHA-256 hash of all proxy .py files, computed at build time by build.rs.
const PROXY_ASSETS_HASH: &str = include_str!(concat!(env!("OUT_DIR"), "/proxy_assets_hash.txt"));

/// Global state for the proxy, used by terminate().
static PROXY_PID: AtomicI32 = AtomicI32::new(0);
static PROXY_PORT: AtomicU16 = AtomicU16::new(0);

/// Gracefully terminate the proxy server.
/// Sends POST /shutdown, then waits for the process to exit.
/// Falls back to SIGKILL after timeout.
pub fn terminate() {
    let port = PROXY_PORT.swap(0, Ordering::Relaxed);
    let pid = PROXY_PID.swap(0, Ordering::Relaxed);

    if port == 0 || pid == 0 {
        return;
    }

    log::info!(
        "[proxy_manager] Sending shutdown request to proxy (port={}, pid={})...",
        port,
        pid
    );

    // Send POST /shutdown to trigger graceful exit.
    let _ = std::net::TcpStream::connect(format!("127.0.0.1:{}", port)).and_then(|mut stream| {
        use std::io::Write;
        let request = format!(
            "POST /shutdown HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nContent-Length: 0\r\n\r\n",
            port
        );
        stream.write_all(request.as_bytes())?;
        stream.set_read_timeout(Some(Duration::from_secs(1)))?;
        let mut buf = [0u8; 256];
        let _ = stream.read(&mut buf);
        Ok(())
    });

    // Wait for process to exit (up to 3 seconds).
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        #[cfg(unix)]
        {
            let result = unsafe { libc::kill(pid, 0) };
            if result != 0 {
                log::info!("[proxy_manager] Proxy server exited gracefully.");
                break;
            }
        }

        if Instant::now() > deadline {
            log::warn!("[proxy_manager] Proxy did not exit in time, force killing (pid={})...", pid);
            #[cfg(unix)]
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    // Clean up env var.
    std::env::remove_var("WARP_PROXY_PORT");
}

/// Called by libc::atexit as a safety net.
#[cfg(unix)]
extern "C" fn kill_proxy_on_exit() {
    let pid = PROXY_PID.swap(0, Ordering::Relaxed);
    if pid > 0 {
        unsafe {
            libc::kill(-pid, libc::SIGKILL);
        }
    }
}

pub struct ProxyManager {
    child: Option<Child>,
    port: u16,
}

impl ProxyManager {
    /// Start the local proxy server. Blocks briefly for venv setup on first run.
    /// Only the primary process should start the proxy — child processes (crash reporter,
    /// plugin host, etc.) skip by checking WARP_PROXY_PORT env var.
    pub fn start() -> anyhow::Result<Self> {
        // If WARP_PROXY_PORT is set, we're a child process — reuse the existing proxy.
        if let Ok(port_str) = std::env::var("WARP_PROXY_PORT") {
            if let Ok(port) = port_str.parse::<u16>() {
                log::info!(
                    "[proxy_manager] Child process: reusing existing proxy on port {}",
                    port
                );
                return Ok(Self {
                    child: None,
                    port,
                });
            }
        }

        let proxy_dir = Self::proxy_dir();
        Self::extract_assets(&proxy_dir)?;
        Self::ensure_venv(&proxy_dir)?;

        let port = Self::find_free_port()?;
        let child = Self::spawn_server(&proxy_dir, port)?;

        // Set env var so child processes know a proxy is already running.
        std::env::set_var("WARP_PROXY_PORT", port.to_string());

        // Store globally for terminate().
        PROXY_PID.store(child.id() as i32, Ordering::Relaxed);
        PROXY_PORT.store(port, Ordering::Relaxed);

        // Register atexit as a safety net.
        #[cfg(unix)]
        unsafe {
            libc::atexit(kill_proxy_on_exit);
        }

        // Wait for the server to be ready.
        Self::wait_for_ready(port)?;
        Ok(Self {
            child: Some(child),
            port,
        })
    }

    /// Poll until the proxy server accepts connections, or timeout.
    fn wait_for_ready(port: u16) -> anyhow::Result<()> {
        use std::net::TcpStream;

        let deadline = Instant::now() + Duration::from_secs(30);
        let addr = format!("127.0.0.1:{}", port);

        loop {
            if TcpStream::connect(&addr).is_ok() {
                log::info!("[proxy_manager] Proxy server is ready on port {}", port);
                return Ok(());
            }
            if Instant::now() > deadline {
                anyhow::bail!(
                    "Proxy server failed to start within 30 seconds on port {}",
                    port
                );
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// The port the proxy server is listening on.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// The full URL to use as server_root_url.
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    fn proxy_dir() -> PathBuf {
        warp_core::paths::data_dir().join("proxy")
    }

    /// Find an available TCP port by binding to port 0.
    fn find_free_port() -> anyhow::Result<u16> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        drop(listener);
        Ok(port)
    }

    fn extract_assets(target: &Path) -> anyhow::Result<()> {
        // Compare the build-time hash of proxy .py files against the on-disk marker.
        // Re-extract only when the hash differs (i.e., source files changed).
        let version = PROXY_ASSETS_HASH.trim();
        let marker = target.join(".version");
        if marker.exists() {
            if let Ok(existing) = fs::read_to_string(&marker) {
                if existing.trim() == version {
                    return Ok(());
                }
            }
        }

        log::info!("[proxy_manager] Extracting proxy assets to {:?}", target);
        fs::create_dir_all(target)?;

        for filename in Assets::iter() {
            let filename_str: &str = filename.as_ref();
            if let Some(rel) = filename_str.strip_prefix("proxy/") {
                if let Some(file) = Assets::get(filename_str) {
                    let dest = target.join(rel);
                    if let Some(parent) = dest.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    fs::write(&dest, file.data.as_ref() as &[u8])?;
                }
            }
        }

        fs::write(&marker, version)?;
        log::info!("[proxy_manager] Extraction complete (hash {})", version);
        Ok(())
    }

    fn ensure_venv(proxy_dir: &Path) -> anyhow::Result<()> {
        let venv = proxy_dir.join(".venv");
        let python_bin = if cfg!(target_os = "windows") {
            venv.join("Scripts").join("python.exe")
        } else {
            venv.join("bin").join("python3")
        };

        if !python_bin.exists() {
            log::info!("[proxy_manager] Creating Python venv...");
            let status = Command::new("python3")
                .args(["-m", "venv", ".venv"])
                .current_dir(proxy_dir)
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .status()?;
            if !status.success() {
                anyhow::bail!("Failed to create Python venv (exit code: {:?})", status.code());
            }
        }

        let pip_bin = if cfg!(target_os = "windows") {
            venv.join("Scripts").join("pip.exe")
        } else {
            venv.join("bin").join("pip")
        };

        log::info!("[proxy_manager] Installing Python dependencies...");
        let status = Command::new(&pip_bin)
            .args(["install", "-q", "-r", "requirements.txt"])
            .current_dir(proxy_dir)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .status()?;
        if !status.success() {
            anyhow::bail!("pip install failed (exit code: {:?})", status.code());
        }

        Self::create_go_features_stub(&venv)?;
        Ok(())
    }

    fn create_go_features_stub(venv: &Path) -> anyhow::Result<()> {
        let site_packages = if cfg!(target_os = "windows") {
            venv.join("Lib")
                .join("site-packages")
                .join("google")
                .join("protobuf")
        } else {
            let lib_dir = venv.join("lib");
            let mut pb_dir = None;
            if let Ok(entries) = fs::read_dir(&lib_dir) {
                for entry in entries.flatten() {
                    let candidate = entry
                        .path()
                        .join("site-packages")
                        .join("google")
                        .join("protobuf");
                    if candidate.exists() {
                        pb_dir = Some(candidate);
                        break;
                    }
                }
            }
            pb_dir.unwrap_or_else(|| {
                lib_dir
                    .join("python3")
                    .join("site-packages")
                    .join("google")
                    .join("protobuf")
            })
        };

        let stub_path = site_packages.join("go_features_pb2.py");
        if !stub_path.exists() {
            let stub_content = r#"# Stub for go_features_pb2 - Go-specific proto extension not needed in Python.
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n!google/protobuf/go_features.proto\x12\x0fgoogle.protobuf'
    b'\x62\x08\x65\x64itionsp\xe8\x07'
)
"#;
            fs::write(&stub_path, stub_content)?;
            log::info!("[proxy_manager] Created go_features_pb2.py stub");
        }
        Ok(())
    }

    fn spawn_server(proxy_dir: &Path, port: u16) -> anyhow::Result<Child> {
        let python = if cfg!(target_os = "windows") {
            proxy_dir.join(".venv/Scripts/python.exe")
        } else {
            proxy_dir.join(".venv/bin/python3")
        };

        log::info!("[proxy_manager] Starting proxy server on port {}...", port);

        // Create a pipe for parent-liveness detection.
        // The parent holds the write end; the child monitors the read end.
        // When the parent exits (even via SIGKILL), the write end closes and
        // the child sees EOF on the read end, triggering a clean exit.
        #[cfg(unix)]
        let parent_alive_read_fd = {
            let mut fds = [0i32; 2];
            let ret = unsafe { libc::pipe(fds.as_mut_ptr()) };
            if ret != 0 {
                anyhow::bail!("Failed to create pipe for parent-liveness detection");
            }
            let read_fd = fds[0]; // child will monitor this
            let _write_fd = fds[1]; // parent keeps this open (leaked intentionally)

            // Set FD_CLOEXEC on the WRITE end so it's NOT inherited by the child.
            // Only the parent should hold the write end.
            unsafe {
                let flags = libc::fcntl(_write_fd, libc::F_GETFD);
                libc::fcntl(_write_fd, libc::F_SETFD, flags | libc::FD_CLOEXEC);
            }

            // Ensure the read end does NOT have CLOEXEC so the child inherits it.
            unsafe {
                let flags = libc::fcntl(read_fd, libc::F_GETFD);
                libc::fcntl(read_fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC);
            }

            // Note: _write_fd is a raw i32, not wrapped in a File/OwnedFd,
            // so it won't be closed when this scope ends. It remains open for
            // the lifetime of this process. When this process exits, the OS
            // closes it, and the child detects EOF.

            read_fd
        };

        let mut cmd = Command::new(&python);
        cmd.args([
            "-m",
            "uvicorn",
            "server:app",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
        ])
        .current_dir(proxy_dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null());

        // Pass the read FD number to the child via environment variable.
        #[cfg(unix)]
        cmd.env("WARP_PARENT_ALIVE_FD", parent_alive_read_fd.to_string());

        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }

        let child = cmd.spawn()?;
        Ok(child)
    }

    /// Check if first-run setup is needed (venv doesn't exist yet).
    pub fn needs_first_run_setup() -> bool {
        let proxy_dir = Self::proxy_dir();
        let venv_python = proxy_dir.join(".venv/bin/python3");
        !venv_python.exists()
    }
}

impl Drop for ProxyManager {
    fn drop(&mut self) {
        // Drop is a no-op — terminate() handles graceful shutdown.
        // This avoids blocking the main thread.
    }
}
