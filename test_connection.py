import asyncio
import websockets

async def test():
    try:
        async with websockets.connect("ws://127.0.0.1:8000/showdown/websocket") as ws:
            print("Connected successfully!")
            msg = await ws.recv()
            print("Received:", msg[:200])
    except Exception as e:
        print("Connection failed:", type(e).__name__, e)

asyncio.run(test())