#!/usr/bin/env python3

import unittest

from render_q36_prompts import render


class RenderTests(unittest.TestCase):
    def test_controls_developer_and_preserved_reasoning(self):
        messages = [
            {"role": "developer", "content": " <|think_off|> policy "},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer", "reasoning_content": "old thought"},
            {"role": "user", "content": " <|think_on|> second "},
        ]
        prompt = render(messages, "think")
        self.assertNotIn("think_off", prompt)
        self.assertNotIn("think_on", prompt)
        self.assertIn("<|im_start|>system\npolicy<|im_end|>", prompt)
        self.assertIn("<think>\nold thought\n</think>\n\nanswer", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n"))

    def test_tool_spacing_and_error_recovery(self):
        calls = [
            {"function": {"name": "one", "arguments": {"x": 1}}},
            {"function": {"name": "two", "arguments": {"y": 2}}},
        ]
        messages = [
            {"role": "user", "content": "run"},
            {"role": "assistant", "content": "", "tool_calls": calls},
            {"role": "tool", "content": "Error: bad argument"},
            {"role": "tool", "content": "failed to open"},
        ]
        prompt = render(messages, "think", tools=[{"type": "function"}])
        self.assertIn("</tool_call>\n\n<tool_call>", prompt)
        self.assertIn("previous tool call returned an error", prompt)
        self.assertIn("Multiple consecutive tool errors", prompt)
        self.assertTrue(prompt.endswith(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_quoted_think_close_is_content(self):
        messages = [
            {"role": "user", "content": "quote"},
            {"role": "assistant", "content": 'The text says "</think>" literally.'},
        ]
        prompt = render(messages, "think")
        self.assertIn('The text says "</think>" literally.', prompt)


if __name__ == "__main__":
    unittest.main()
