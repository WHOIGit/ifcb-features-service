from stateless_microservice import ServiceConfig, create_app

from .service import FeatureProcessor

config = ServiceConfig(
    description="Low-latency blob extraction from base64-encoded IFCB ROI png.",
)

app = create_app(FeatureProcessor(), config)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010)