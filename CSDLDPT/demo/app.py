"""
Redirect stub — main demo đã chuyển sang app/demo.py
Chạy: python app/demo.py
"""
import os
import sys

print("⚠️  Demo đã chuyển sang app/demo.py")
print("   Chạy: python app/demo.py")

# Auto-redirect
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
exec(open(os.path.join(os.path.dirname(__file__), '..', 'app', 'demo.py')).read())
