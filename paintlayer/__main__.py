"""Waitress entry point for local launches and Compose."""

from waitress import serve

from .settings import Runtime
from .web import create_app


def main():
    runtime = Runtime.from_environment()
    app = create_app(runtime)
    print(f'PaintLayer Desk listening at http://{runtime.host}:{runtime.port}', flush=True)
    serve(
        app,
        host=runtime.host,
        port=runtime.port,
        threads=runtime.threads,
        channel_timeout=60,
        max_request_body_size=app.config['MAX_CONTENT_LENGTH'],
        ident='PaintLayer',
    )


if __name__ == '__main__':
    main()
