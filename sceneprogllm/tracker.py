import json

class UsageTracker:
    def __init__(self):
        self.calls = []

    def record(self, usage, name, response_format):
        entry = {
            "call": len(self.calls) + 1,
            "name": name,
            "response_format": response_format,
            **usage,
        }
        self.calls.append(entry)

    @property
    def aggregate(self):
        return {
            "total_calls": len(self.calls),
            "total_prompt_tokens": sum(c.get("prompt_tokens", 0) for c in self.calls),
            "total_completion_tokens": sum(c.get("completion_tokens", 0) for c in self.calls),
            "total_tokens": sum(c.get("total_tokens", 0) for c in self.calls),
            "total_cost_usd": round(sum(c.get("cost_usd", 0) for c in self.calls), 8),
            "total_latency_s": round(sum(c.get("latency_s", 0) for c in self.calls), 4),
        }

    def export(self, path):
        data = {"aggregate": self.aggregate, "calls": self.calls}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return data
