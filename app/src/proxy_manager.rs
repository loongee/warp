//! Manages the embedded Python proxy server for local DeepSeek AI conversations.
//!
//! On startup:
//! 1. Extracts embedded proxy Python source to ~/.warp-proxy/
//! 2. Creates a Python venv and installs dependencies (first run only)
//! 3. Finds a free port dynamically
//! 4. Spawns uvicorn as a child process on that port
//! 5. Kills the child when dropped (Warp exits)

use std::fs;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use crate::Assets;

pub struct ProxyManager {
    child: Option<Child>,
    port: u16,
}

impl ProxyManager {
    /// Start the local proxy server. Blocks briefly for venv setup on first run.
    /// Returns the manager which holds the child process and the allocated port.
    pub fn start() -> anyhow::Result<Self> {
        let proxy_dir = Self::proxy_dir();
        Self::extract_assets(&proxy_dir)?;
        Self::ensure_venv(&proxy_dir)?;

        let port = Self::find_free_port()?;
        let child = Self::spawn_server(&proxy_dir, port)?;

        // Wait for the server to be ready (up to 30 seconds for first-run import).
        Self::wait_for_ready(port)?;
        Ok(Self {
            child: Some(child),
            port,
        })
    }

    /// Poll until the proxy server accepts connections, or timeout.
    fn wait_for_ready(port: u16) -> anyhow::Result<()> {
        use std::net::TcpStream;
        use std::time::{Duration, Instant};

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
        dirs::home_dir()
            .expect("Could not determine home directory")
            .join(".warp-proxy")
    }

    /// Find an available TCP port by binding to port 0.
    fn find_free_port() -> anyhow::Result<u16> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        // Drop the listener to release the port before uvicorn binds it.
        drop(listener);
        Ok(port)
    }

    fn extract_assets(target: &Path) -> anyhow::Result<()> {
        // Use CARGO_PKG_VERSION as a version marker to avoid re-extracting unchanged files.
        let version = env!("CARGO_PKG_VERSION");
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
        log::info!("[proxy_manager] Extraction complete (version {})", version);
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

        // Install/update dependencies. pip install -q is fast when already satisfied.
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

        // Create go_features_pb2.py stub in the protobuf package.
        // The generated proto code imports this Go-specific extension that isn't
        // shipped with the Python protobuf package.
        Self::create_go_features_stub(&venv)?;

        Ok(())
    }

    fn create_go_features_stub(venv: &Path) -> anyhow::Result<()> {
        // Find the google.protobuf package directory inside the venv.
        let site_packages = if cfg!(target_os = "windows") {
            venv.join("Lib").join("site-packages")
        } else {
            // Find the python version directory dynamically.
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

        log::info!(
            "[proxy_manager] Starting proxy server on port {}...",
            port
        );

        let child = Command::new(&python)
            .args([
                "-m",
                "uvicorn",
                "server:app",
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
            ])
            .current_dir(proxy_dir)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()?;

        Ok(child)
    }
}

impl Drop for ProxyManager {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            log::info!("[proxy_manager] Shutting down proxy server...");
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
