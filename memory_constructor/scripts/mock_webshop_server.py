#!/usr/bin/env python3
"""
Mock WebShop Pool Server for pipeline testing.

Simulates the WebShop environment with deterministic behavior
when the real WebShop (Java + SimServer) is unavailable.

API matches webshop_server_pool.py:
  POST /create_session  -> {session_id, obs, instruction_text}
  POST /step            -> {obs, reward, done, info}
  POST /close_session   -> {status: "ok"}
  GET  /health          -> {status: "ok", active_sessions: N}

Usage:
    python mock_webshop_server.py --port 6001
"""

import uuid
import random
import threading
import argparse
import logging
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample product data for mock environment
PRODUCTS = [
    {"id": "B09QQWRW2M", "name": "Red Laptop Sleeve 15.6 inch", "price": 19.99,
     "attrs": "Color: red | Size: 15.6 inch | Material: neoprene"},
    {"id": "B08N5WRWNW", "name": "Wireless Bluetooth Headphones", "price": 39.99,
     "attrs": "Color: black | Type: over-ear | Battery: 30h"},
    {"id": "B07VGRJDFY", "name": "Stainless Steel Water Bottle 32oz", "price": 24.99,
     "attrs": "Color: silver | Capacity: 32oz | Insulated: yes"},
    {"id": "B09H3LPWQX", "name": "USB-C Hub Adapter 7-in-1", "price": 29.99,
     "attrs": "Ports: HDMI, USB3, SD | Compatible: laptop"},
    {"id": "B0BT8GLSLP", "name": "Cotton T-Shirt Pack of 3", "price": 25.00,
     "attrs": "Color: white/black/gray | Size: M | Material: 100% cotton"},
]

TASKS = [
    "I need a red laptop sleeve that fits a 15.6 inch laptop, and price lower than 30.00 dollars",
    "Find me wireless headphones in black color with at least 20 hours battery life, under 50 dollars",
    "Looking for a large insulated water bottle, at least 32 ounces, under 30 dollars",
    "I want a USB-C hub adapter with HDMI port for my laptop, price under 40 dollars",
    "Buy a pack of cotton t-shirts in medium size, under 30 dollars",
]


class MockSession:
    def __init__(self, task_idx=None):
        self.task_idx = task_idx if task_idx is not None else random.randint(0, len(TASKS) - 1)
        self.task_idx = self.task_idx % len(TASKS)
        self.instruction = TASKS[self.task_idx]
        self.target_product = PRODUCTS[self.task_idx]
        self.step_count = 0
        self.page = "search"  # search, results, product, done
        self.searched = False
        self.clicked_product = False

    def get_initial_obs(self):
        return (
            "WebShop [SEP] "
            "Instruction: [SEP] "
            f"{self.instruction} [SEP] "
            "[Search]"
        )

    def step(self, action):
        self.step_count += 1
        action_lower = action.lower().strip()

        if "search[" in action_lower:
            self.page = "results"
            self.searched = True
            # Show search results
            products_text = []
            for p in PRODUCTS:
                products_text.append(
                    f"[Back to Search] [SEP] "
                    f"Page 1 (Total results: {len(PRODUCTS)}) [SEP] "
                    f"[Next >] [SEP] "
                )
            results = []
            for p in PRODUCTS:
                results.append(f"{p['name']} [SEP] {p['price']} [SEP] [{p['id']}]")
            obs = (
                f"Instruction: [SEP] {self.instruction} [SEP] "
                f"[Back to Search] [SEP] Page 1 (Total results: {len(PRODUCTS)}) [SEP] [Next >] [SEP] "
                + " [SEP] ".join(results)
            )
            return obs, 0.0, False, {}

        elif "click[" in action_lower:
            inner = action.split("[")[1].rstrip("]").strip()

            if inner.lower() == "buy now":
                self.page = "done"
                # Compute reward based on whether correct product was selected
                reward = 1.0 if self.clicked_product else 0.1
                obs = "Thank you for shopping! Your order has been placed."
                return obs, reward, True, {"reward": reward}

            elif inner.lower() == "back to search":
                self.page = "search"
                obs = (
                    "WebShop [SEP] "
                    f"Instruction: [SEP] {self.instruction} [SEP] "
                    "[Search]"
                )
                return obs, 0.0, False, {}

            elif inner.upper().startswith("B0"):
                # Clicked a product ID
                self.page = "product"
                # Check if it's the target product
                if inner.upper() == self.target_product["id"]:
                    self.clicked_product = True
                    p = self.target_product
                else:
                    p = random.choice(PRODUCTS)

                obs = (
                    f"Instruction: [SEP] {self.instruction} [SEP] "
                    f"[Back to Search] [SEP] [< Prev] [SEP] "
                    f"{p['name']} [SEP] "
                    f"Price: ${p['price']} [SEP] "
                    f"{p['attrs']} [SEP] "
                    f"[Buy Now]"
                )
                return obs, 0.0, False, {}

            else:
                # Unknown click target
                obs = f"Invalid click target: {inner}. Available actions: search[...] or click[product_id/buy now/back to search]"
                return obs, 0.0, False, {}

        else:
            obs = "Invalid action format. Use search[keywords] or click[target]."
            return obs, 0.0, False, {}


class MockWebShopPool:
    def __init__(self):
        self._sessions = {}
        self._lock = threading.Lock()

    def create_session(self, task_idx=None):
        session_id = str(uuid.uuid4())
        session = MockSession(task_idx)
        with self._lock:
            self._sessions[session_id] = session
        obs = session.get_initial_obs()
        return session_id, obs, session.instruction

    def step(self, session_id, action):
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session {session_id} not found")
        obs, reward, done, info = session.step(action)
        if done:
            with self._lock:
                self._sessions.pop(session_id, None)
        return obs, reward, done, info

    def close_session(self, session_id):
        with self._lock:
            self._sessions.pop(session_id, None)

    @property
    def active_count(self):
        with self._lock:
            return len(self._sessions)


def create_app():
    app = Flask(__name__)
    pool = MockWebShopPool()

    @app.route("/create_session", methods=["POST"])
    def create_session():
        data = request.get_json(force=True, silent=True) or {}
        task_idx = data.get("task_idx")
        session_id, obs, instruction = pool.create_session(task_idx)
        return jsonify({"session_id": session_id, "obs": obs, "instruction_text": instruction})

    @app.route("/step", methods=["POST"])
    def step():
        data = request.get_json(force=True, silent=True) or {}
        session_id = data.get("session_id", "")
        action = data.get("action", "")
        try:
            obs, reward, done, info = pool.step(session_id, action)
            return jsonify({"obs": obs, "reward": reward, "done": done, "info": info})
        except KeyError as e:
            return jsonify({"error": str(e)}), 404

    @app.route("/close_session", methods=["POST"])
    def close_session():
        data = request.get_json(force=True, silent=True) or {}
        pool.close_session(data.get("session_id", ""))
        return jsonify({"status": "ok"})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "active_sessions": pool.active_count, "mock": True})

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=6001)
    args = parser.parse_args()

    app = create_app()
    logger.info(f"Starting MOCK WebShop pool server on port {args.port}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)
