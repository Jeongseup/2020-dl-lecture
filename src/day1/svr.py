#!/usr/bin/env python3
import sys
from http.server import HTTPServer, CGIHTTPRequestHandler, test

# python -m http.server --cgi 8000
if __name__ == '__main__':
    test(CGIHTTPRequestHandler, HTTPServer, port=int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
