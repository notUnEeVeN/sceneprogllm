from sceneprogllm import LLM, UsageTracker
import json

tracker = UsageTracker()

llm1 = LLM(response_format="text", name="summarizer", tracker=tracker)
llm2 = LLM(response_format="list", name="lister", tracker=tracker)

r1 = llm1("What is the capital of France?")
r2 = llm2("List 3 primary colors")

print("=== Results ===")
print("summarizer:", r1)
print("lister:", r2)

print("\n=== Individual call stats ===")
print(json.dumps(tracker.calls, indent=2))

print("\n=== Aggregate ===")
print(json.dumps(tracker.aggregate, indent=2))

tracker.export("usage.json")
print("\nExported to usage.json")
