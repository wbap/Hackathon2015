from __future__ import print_function
import sys
import socket
from contextlib import closing

def main():
  host = '127.0.0.1'
  port = 2000
  bufsize = 40960

  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  with closing(sock):
    sock.connect((host, port))
    while True:
      line = sys.stdin.readline().rstrip()
      if len(line) == 0:
        break
      sock.send(line.encode('utf-8'))
      print(sock.recv(bufsize))
  return

if __name__ == '__main__':
  main()
