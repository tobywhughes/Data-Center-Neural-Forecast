"""
Tobias Hughes

Converts PCAP files and converts them into a time series format. 
Tested and optimized to be used with the CAIDA Anonymized data set.
"""
import dpkt
import matplotlib.pyplot as plt
import os
import pickle

def import_pcap_list(filename, print_num = 3000000, print_progress = True):
    print("Parsing file: ", filename)
    if exists_in_cache(filename):
        return read_pcap_cache(filename)
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
    cache_pcap_import(packet_list, filename)
    f.close()
    return packet_list

def cache_pcap_import(packet_list, filename):
    stripped_name = os.path.basename(filename).rstrip(".anon.pcap")
    pickle_name = '.\\cache\\' +stripped_name + '.p'
    os.makedirs(os.path.dirname(pickle_name), exist_ok=True)
    pickle.dump(packet_list, open(pickle_name, 'wb'))
    print("Cached in ", os.path.abspath(pickle_name))

def exists_in_cache(filename):
    stripped_name = os.path.basename(filename).rstrip(".anon.pcap")
    pickle_name = '.\\cache\\' +stripped_name + '.p'
    os.makedirs(os.path.dirname(pickle_name), exist_ok=True)
    return os.path.isfile(pickle_name)

def read_pcap_cache(filename):
    stripped_name = os.path.basename(filename).rstrip(".anon.pcap")
    pickle_name = '.\\cache\\' +stripped_name + '.p'
    print("Read in cache from", os.path.abspath(pickle_name))
    return pickle.load(open(pickle_name, 'rb'))


def convert_to_seconds_series(packet_list):
    current_packet_count = 0
    last_packet_time = 0
    packet_times = []
    start_time = packet_list[0]
    update_flag = False
    for packet in packet_list:
        current_packet_time = packet - start_time
        if current_packet_time == last_packet_time:
            current_packet_count += 1
            update_flag = False
        else:
            packet_times.append((packet - 1, current_packet_count))
            current_packet_count = 1
            update_flag == True
        last_packet_time = current_packet_time
    if update_flag == False:
        packet_times.append((packet_list[-1], current_packet_count))
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
    f.close()

def output_series_by_line(packet_list):
    for times in packet_times:
        print("Time: ", times[0], "  Count: ", times[1])

def run_pcap_imports(filename_list):
    packet_lists = []
    for filename in filename_list:
        packet_lists.append(import_pcap_list(filename))
    return packet_lists

def flatten_list(packet_lists):
    packet_list = []
    for current_list in packet_lists:
        packet_list += current_list
    return  sorted(packet_list)

def show_time_series(packet_list):
    times = [second[0] for second in packet_list]
    counts =[count[1] for count in packet_list]
    plt.plot(times, counts)
    plt.axis([0, len(times), 0, (max(counts) * 1.1)])
    plt.show()

if __name__ == "__main__":
    name_list = read_list_of_names("./pcap_list.txt")
    packet_lists = run_pcap_imports(name_list)
    packet_list = flatten_list(packet_lists)
    packet_times = move_to_zero(convert_to_seconds_series(packet_list))
    padded_packet_times = zero_pad(packet_times, 180)
    #output_series_by_line(padded_packet_times)
    show_time_series(padded_packet_times)