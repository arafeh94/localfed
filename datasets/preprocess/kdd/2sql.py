import json
import mysql.connector

db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='root',
    database='kdd'
)
cursor = db.cursor()

protocol_type = ['tcp', 'udp', 'icmp']
service = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo',
           'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http',
           'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link',
           'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u',
           'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp',
           'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i',
           'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
flag = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
clazz = ['normal', 'anomaly']
file = open('../data/kdd/KDDTrain+.arff', 'r')
xs = []
ys = []
while True:
    line = file.readline()
    if not line:
        break
    if line.startswith('@'):
        continue
    line = line.replace('\n', '')
    data = line.split(',')
    data[1] = protocol_type.index(data[1])
    data[2] = service.index(data[2])
    data[3] = flag.index(data[3])
    data[-1] = clazz.index(data[-1])
    floats = [float(dt) for dt in data]
    xs.append(floats[:-1])
    ys.append(floats[-1])

for x, y in zip(xs, ys):
    query = "insert into kdd.sample values (null, %s, %s, %s, %s)"
    cursor.execute(query, (0, json.dumps(x), int(y), 0))
db.commit()
