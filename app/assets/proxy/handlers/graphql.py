"""GraphQL mock handler for Warp client compatibility."""
import json
import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("warp-proxy.graphql")

router = APIRouter()

# A single model entry reusable across all feature categories.
DEEPSEEK_MODEL_ENTRY = {
    "id": "deepseek-chat",
    "displayName": "DeepSeek V3",
    "baseModelName": "deepseek-chat",
    "reasoningLevel": None,
    "provider": "Openai",
    "visionSupported": False,
    "description": "DeepSeek V3 via local proxy",
    "disableReason": None,
    "spec": {"cost": 0.1, "quality": 0.9, "speed": 0.8},
    "usageMetadata": {
        "requestMultiplier": 1,
        "creditMultiplier": None,
    },
    "pricing": {
        "discountPercentage": None,
    },
    "hostConfigs": [],
}

AVAILABLE_LLMS = {
    "defaultId": "deepseek-chat",
    "choices": [DEEPSEEK_MODEL_ENTRY],
    "preferredCodexModelId": None,
}

FEATURE_MODEL_CHOICE = {
    "agentMode": AVAILABLE_LLMS,
    "coding": AVAILABLE_LLMS,
    "cliAgent": AVAILABLE_LLMS,
    "computerUseAgent": AVAILABLE_LLMS,
}


def _model_choices_response():
    """GetFeatureModelChoices / workspace-level model list."""
    return {
        "data": {
            "user": {
                "__typename": "UserOutput",
                "user": {
                    "workspaces": [
                        {
                            "uid": "local-workspace",
                            "featureModelChoice": FEATURE_MODEL_CHOICE,
                        }
                    ],
                },
            }
        }
    }


def _free_models_response():
    """FreeAvailableModels — no auth required."""
    return {
        "data": {
            "freeAvailableModels": {
                "__typename": "FreeAvailableModelsOutput",
                "featureModelChoice": FEATURE_MODEL_CHOICE,
            }
        }
    }


def _request_limit_response():
    """GetRequestLimitInfo — unlimited quota."""
    return {
        "data": {
            "user": {
                "__typename": "UserOutput",
                "user": {
                    "requestLimitInfo": {
                        "limit": 999999,
                        "numRequestsUsedSinceRefresh": 0,
                        "nextRefreshTime": "2099-01-01T00:00:00Z",
                        "isUnlimited": True,
                    },
                    "workspaces": [],
                    "bonusGrants": [],
                },
            }
        }
    }


# Map known operation names to response builders.
OPERATION_HANDLERS = {
    "GetFeatureModelChoices": _model_choices_response,
    "FreeAvailableModels": _free_models_response,
    "GetRequestLimitInfo": _request_limit_response,
}


@router.post("/graphql/v2")
async def graphql_handler(request: Request):
    """Handle GraphQL requests by dispatching on the `op` query parameter."""
    op = request.query_params.get("op", "")

    handler = OPERATION_HANDLERS.get(op)
    if handler:
        return JSONResponse(handler())

    # For any other operation, try to extract from request body.
    try:
        body = await request.json()
        op_name = body.get("operationName", op)
        handler = OPERATION_HANDLERS.get(op_name)
        if handler:
            return JSONResponse(handler())
    except Exception:
        pass

    # Fallback: return empty data so the client doesn't crash.
    logger.debug("Unhandled operation: %s", op)
    return JSONResponse({"data": {}})
