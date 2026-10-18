import argparse
from services.inference.play_api import PlayApiServer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ZenZero Play Server")
    parser.add_argument('--port', type=int, default=5000, help="Listen port")
    args = parser.parse_args()

    server = PlayApiServer()
    server.run(port=args.port)
