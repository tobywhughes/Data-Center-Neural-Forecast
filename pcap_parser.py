"""
Tobias Hughes

Converts PCAP files and converts them into a time series format. 
Tested and optimized to be used with the CAIDA Anonymized data set.
"""
import dpkt

def import_pcap_list(filename, print_num = 3000000, print_progress = True):
    print("Parsing file: ", filename)
    f = open(filename, mode = 'rb')
    pcap = dpkt.pcap.Reader(f)
    packets = 0
    packet_list = []
    for packet in pcap:
        packets += 1
        packet_time = int(packet[0])
        packet_list.append(packet_time)
        if packets % print_num == 0 and print_progress:
            print("Packets Processed: ", packets)
    print("Total Packets Processed: ", packets)
    f.close()
    return packet_list

def convert_to_seconds_series(packet_list):
    current_packet_count = 0
    last_packet_time = 0
    packet_times = []
    start_time = packet_list[0]
    for packet in packet_list:
        current_packet_time = packet - start_time
        if current_packet_time == last_packet_time:
            current_packet_count += 1
        else:
            packet_times.append((packet - 1, current_packet_count))
            current_packet_count = 1
        last_packet_time = current_packet_time
    return packet_times

def move_to_zero(packet_list):
    converted_list = []
    start_time = packet_list[0][0]
    for packet in packet_list:
        converted_list.append((packet[0] - start_time, packet[1]))
    return converted_list

def zero_pad(packet_list, end_time):
    if (end_time) < len(packet_list):
        print("Warning - End time less than size of list.")
    i = 0
    while i < end_time:
        if len(packet_times) <= i:
            packet_times.insert(i, (i, 0))
        elif packet_times[i][0] != i:
            packet_times.insert(i, (i, 0))
        i += 1
    return packet_list

def read_list_of_names(filename):
    filename_list = []
    f = open(filename, 'r')
    for line in f:
        if line[0] != '#':
            filename_list.append(line.rstrip())
    return filename_list

def output_series_by_line(packet_list):
    for times in packet_times:
        print("Time: ", times[0], "  Count: ", times[1])

if __name__ == "__main__":
    name_list = read_list_of_names("./pcap_list.txt")
    packet_list = import_pcap_list(name_list[0])
    packet_list2 = import_pcap_list(name_list[1])
    packet_times = move_to_zero(convert_to_seconds_series(packet_list))
    padded_packet_times = zero_pad(packet_times, 60)
    output_series_by_line(padded_packet_times)