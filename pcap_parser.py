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
    print("Converting pcap series to time series....")
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
    "Converting time series to 0 offset..."
    converted_list = []
    start_time = packet_list[0][0]
    for packet in packet_list:
        converted_list.append((packet[0] - start_time, packet[1]))
    return converted_list

def zero_pad(packet_list):
    print("Zero padding time series....")
    i = 0
    while i < packet_list[-1][0]:
        if packet_list[i][0] != i:
            packet_list.insert(i, (i, 0))
        i += 1
    return packet_list

def read_list_of_names(filename):
    filename_list = []
    f = open(filename, 'r')
    for line in f:
        if line[0] != '#':
            filename_list.append(line.rstrip())
    f.close()
    return filename_list

def output_series_by_line(packet_list):
    for times in packet_list:
        print("Time: ", times[0], "  Count: ", times[1])

def run_pcap_imports(filename_list):
    packet_lists = []
    for filename in filename_list:
        packet_lists.append(import_pcap_list(filename))
    return packet_lists

def flatten_list(packet_lists):
    print("Concatenating imports...")
    packet_list = []
    for current_list in packet_lists:
        packet_list += current_list
    return  sorted(packet_list)

def show_time_series(packet_list, subsets):
    print("Plotting Data")
    times = [second[0] for second in packet_list]
    counts =[count[1] for count in packet_list]
    plt.plot(times, counts)
    plt.axis([0, len(times), 0, (max(counts) * 1.1)])
    plt.show()
    subset_times_list = [[second[0] for second in subset] for subset in subsets]
    subset_count_list = [[count[1] for count in subset] for subset in subsets]
    for i in range(3):
        plt.plot(subset_times_list[i*150], subset_count_list[i*150])
        plt.show()

def check_for_time_series_cache(filename_list):
    print('Checking for time series cache...')
    time_series_pname = '.\\cache\\series_cache\\series.p'
    filename_list_pname = '.\\cache\\series_cache\\filenames.p'
    subset_pname = '.\\cache\\series_cache\\subset.p'
    os.makedirs(os.path.dirname(time_series_pname), exist_ok=True)
    if os.path.isfile(time_series_pname) and os.path.isfile(filename_list_pname) and os.path.isfile(subset_pname):
        if pickle.load(open(filename_list_pname, 'rb')) == filename_list:
            return True
    return False

def read_time_series_cache():
    print("Reading time series cache...")
    time_series_pname = '.\\cache\\series_cache\\series.p'
    return pickle.load(open(time_series_pname, 'rb'))

def cache_time_series(time_series, filename_list):
    print("Caching time series...")
    time_series_pname = '.\\cache\\series_cache\\series.p'
    filename_list_pname = '.\\cache\\series_cache\\filenames.p'
    os.makedirs(os.path.dirname(time_series_pname), exist_ok=True)
    pickle.dump(time_series, open(time_series_pname, 'wb'))
    print("Cached in ", os.path.abspath(time_series_pname))
    pickle.dump(filename_list, open(filename_list_pname, 'wb'))
    print("Cached in ", os.path.abspath(filename_list_pname))

def cache_subsets(subset_list):
    print("Caching subseries...")
    subset_pname = '.\\cache\\series_cache\\subset.p'
    os.makedirs(os.path.dirname(subset_pname), exist_ok=True)
    pickle.dump(subset_list, open(subset_pname, 'wb'))
    print("Cached in ", os.path.abspath(subset_pname))

def read_subset_cache():
    print("Reading subset cache...")
    subset_pname = '.\\cache\\series_cache\\subset.p'
    return pickle.load(open(subset_pname, 'rb'))

def generate_time_series(packet_lists):
    print("Generating time series...")
    packet_list = flatten_list(packet_lists)
    packet_times = move_to_zero(convert_to_seconds_series(packet_list))
    return zero_pad(packet_times)

def parse_subsets(packet_list, window_size):
    print("Parsing subsets...")
    current_index = 0
    subset_list = []
    while (current_index + window_size) <= len(packet_list):
        subset_list.append(move_to_zero(packet_list[current_index:(current_index + window_size)]))
        current_index += 1
    return subset_list

if __name__ == "__main__":
    name_list = read_list_of_names("./pcap_list.txt")
    padded_packet_times = []
    subset_list = []
    if check_for_time_series_cache(name_list):
        padded_packet_times = read_time_series_cache()
        subset_list = read_subset_cache()
    else:
        packet_lists = run_pcap_imports(name_list)
        padded_packet_times = generate_time_series(packet_lists)
        cache_time_series(padded_packet_times, name_list)
        subset_list = parse_subsets(padded_packet_times, 240 + 64)
        cache_subsets(subset_list)
    show_time_series(padded_packet_times, subset_list)