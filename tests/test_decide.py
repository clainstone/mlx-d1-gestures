import http.client
import json
import os
import socket
import threading
import unittest
from unittest.mock import patch

import decide


class PortTests(unittest.TestCase):
    def test_occupied_default_uses_free_loopback_port(self):
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            port = occupied.getsockname()[1]
            with patch.dict(os.environ), patch.object(decide, "DEFAULT_PORT", port):
                os.environ.pop("DECIDE_PORT", None)
                with decide.bind_server() as server:
                    self.assertEqual(server.server_address[0], "127.0.0.1")
                    self.assertNotEqual(server.server_port, port)
                os.environ["DECIDE_PORT"] = str(port)
                with self.assertRaisesRegex(SystemExit, f"Port {port} is in use"):
                    decide.bind_server()

    def test_invalid_explicit_port_exits_cleanly(self):
        for port in ("garbage", "-1", "65536"):
            with self.subTest(port=port), patch.dict(os.environ, DECIDE_PORT=port):
                with self.assertRaisesRegex(SystemExit, "DECIDE_PORT must be a port number"):
                    decide.bind_server()


class RequestTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, DECIDE_PORT="0"):
            self.server = decide.bind_server()
        thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02})
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(self.server.shutdown)
        self.client = http.client.HTTPConnection(*self.server.server_address, timeout=2)
        self.addCleanup(self.client.close)

    def test_invalid_body_length_receives_json_error(self):
        for length, code in ((None, 411), ("abc", 400), ("-1", 413), (str(decide.MAX_FRAME + 1), 413)):
            with self.subTest(length=length):
                self.client.putrequest("POST", "/run")
                if length is not None:
                    self.client.putheader("Content-Length", length)
                self.client.endheaders()
                with self.client.getresponse() as response:
                    self.assertEqual(response.status, code)
                    self.assertTrue(json.load(response)["error"])

    def test_truncated_body_leaves_server_responsive(self):
        self.client.request("POST", "/run", body=b"x", headers={"Content-Length": "2"})
        self.client.sock.shutdown(socket.SHUT_WR)
        with self.client.getresponse() as response:
            self.assertEqual(response.status, 400)
            self.assertEqual(json.load(response)["error"], "Incomplete request body")
        self.client.request("GET", "/status")
        with self.client.getresponse() as response:
            self.assertEqual(response.status, 200)

    def test_game_mode_follows_active_questions(self):
        reordered = dict(reversed(list(decide.QUESTIONS.items())))
        for questions, expected in ((reordered, True), (decide.GESTURE_QUESTIONS, False)):
            with self.subTest(game=expected), patch.object(decide, "questions", questions):
                self.client.request("GET", "/questions")
                with self.client.getresponse() as response:
                    self.assertEqual(response.status, 200)
                    payload = json.load(response)
                self.assertEqual(list(payload["questions"]), list(questions))
                self.assertIs(payload["rps"], expected)

    def test_invalid_questions_do_not_replace_active_set(self):
        original = decide.questions
        self.client.request("POST", "/questions", body=b'{"questions": []}')
        with self.client.getresponse() as response:
            self.assertEqual(response.status, 400)
            self.assertTrue(json.load(response)["error"])
        self.assertIs(decide.questions, original)


class MachineSampleTests(unittest.TestCase):
    def test_multiline_array_yields_every_sample(self):
        metrics = '{"cpu_usage":13.4,"soc_metrics":{"gpu_active":83.7,"ane_active":0,"total_power":19.14}}'
        expected = {"cpu": 13, "gpu": 84, "ane": 0, "watts": 19.1}
        self.assertEqual(decide.machine_sample("[" + metrics), expected)
        self.assertEqual(decide.machine_sample("," + metrics), expected)
        self.assertIsNone(decide.machine_sample("]"))


if __name__ == "__main__":
    unittest.main()
