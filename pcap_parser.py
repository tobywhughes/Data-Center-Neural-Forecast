"""
Tobias Hughes

Converts PCAP files and converts them into a time series format. 
Tested and optimized to be used with the CAIDA Anonymized data set.
"""
import dpkt

f = open("C:\\Users\\Toby\\Documents\\temp\\equinix-sanjose.dirA.20121018-125905.UTC.anon.pcap", mode='rb')
pcap = dpkt.pcap.Reader(f)
packets = 0
current_packet_count = 0
packet_list = []
for packet in pcap:
    packets += 1
    packet_time = int(packet[0])
    packet_list.append(packet_time)
    if packets % 3000000 == 0:
        print(packets)

start_time = packet_list[0]
print (start_time)
print(packet_list[0])
last_packet_time = 0
packet_times = []
for packet in packet_list:
    current_packet_time = packet - start_time    
    if current_packet_time == last_packet_time:
        current_packet_count += 1
    else:
        packet_times.append((current_packet_time - 1, current_packet_count))
        current_packet_count = 1

    last_packet_time = current_packet_time

i = 0
while i < 60:
    if len(packet_times) <= i:
        packet_times.insert(i, (i, 0))
    elif packet_times[i][0] != i:
        packet_times.insert(i, (i, 0))
    i += 1

print("Packets: ", packets)
for times in packet_times:
    print("Time: ", times[0], "  Count: ", times[1])

f.close()

if __name__ == "__main__":
    pass