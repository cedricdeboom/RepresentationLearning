#!/usr/bin/python

## USAGE
##  ./find_dir_language.py dir_name
##

__author__ = 'cedricdeboom'

import sys
import os
import subprocess
import paramiko
import scp

def convert_file(file_name, output_file_name):
    g = open(output_file_name, 'w')

    with open(file_name, 'r') as f:
        temp = ''
        for line in f:
            if not line.endswith("\";\n"):
                temp += line + ' '
                continue
            temp += line
            #Example line: id="516577332946288641";cr="1411996608380";text="bla bla bla";hsht="";...;
            indexID = temp.index('id=\"',0)
            indexID2 = temp.index('\";cr=\"', indexID+1)
            id = temp[indexID+4:indexID2]
            indexCR2 = temp.index('\";text=\"', indexID2+1)
            indexText2 = temp.index('\";hsht=\"', indexCR2+1)
            text = temp[indexCR2+8:indexText2]
            output = id + "\t...\t" + text + "\n"
            g.write(output)
            temp = ''

    g.close()

def process_lang_file(lang_file_name, output_file_name):
    g = open(output_file_name, 'w')

    with open(lang_file_name, 'r') as f:
        for line in f:
            #Example line: 	en	554049200263274496	...	lets go half on the sun.
            if len(line) < 5:
                continue
            if line.startswith('>'):
                continue
            tokens = line.split('\t')
            output = tokens[1] + "\t" + tokens[2] + "\n"
            g.write(output)

    g.close()

def convert_dir(dir_name):
    for file_name in os.listdir(dir_name):
        if file_name.endswith('.lang.txt') or file_name.endswith('.temp.txt'):
            continue

        absolute_file = os.path.join(dir_name, file_name)
        temp_file = os.path.join(dir_name, file_name.replace('.txt', '.temp.txt'))
        lang_temp_file = os.path.join(dir_name, file_name.replace('.txt', '.temp.lang.txt'))
        lang_file = os.path.join(dir_name, file_name.replace('.txt', '.lang.txt'))

        if os.path.isfile(lang_file):
            continue

        print "Converting " + absolute_file
        convert_file(absolute_file, temp_file)

        print "Detecting languages"
        subprocess.call("ldig/ldig.py -m ldig/models/model.latin " + temp_file + " >> " + lang_temp_file, shell=True)

        print "Post processing and cleanup"
        process_lang_file(lang_temp_file, lang_file)
        os.remove(temp_file)
        os.remove(lang_temp_file)

def convert_remote_dir(dir_name, log_file_name):
    #get file list in remote dir
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.load_system_host_keys()
        client.connect('157.193.215.41', username='cedric', password='v6SaMhg8SDxvxUfL')
        _, chan_out, chan_err = client.exec_command('ls ' + dir_name)
        out, err = give_out_err(chan_out, chan_err)
        client.close()
    except paramiko.SSHException:
        print "SSH Exception while retrieving ls dir for " + dir_name
        write_log("SSH Exception while retrieving ls dir for " + dir_name + "\n", log_file_name)

    if len(err) > 0:
        print "Could not retrieve ls dir for " + dir_name
        write_log("Could not retrieve ls dir for " + dir_name + "\n", log_file_name)

    file_list = out.split('\n')[0:-1]
    subprocess.call("mkdir " + dir_name, shell=True)

    for file_name in file_list:
        convert_remote_file(file_name, dir_name, log_file_name)

def convert_remote_file(file_name, dir_name, log_file_name):
    #get file from remote
    absolute_file = os.path.join(dir_name, file_name)
    print "Retrieving " + absolute_file
    tries = 2
    client = None
    c = False
    while tries > 0:
        try:
            print 'Try ' + str(tries)
            client = paramiko.SSHClient()
            print 'Client created'
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.load_system_host_keys()
            client.connect('157.193.215.41', username='cedric', password='v6SaMhg8SDxvxUfL')
            print 'Client connected'
            scp_client = scp.SCPClient(client.get_transport())
            print 'scp initiated'
            scp_client.get(absolute_file, dir_name)
            print 'scp succeeded'
            scp_client.close()
            client.close()
            tries = 0
        except:
            print "SCP Exception while retrieving file " + file_name
            write_log("SCP Exception while retrieving file " + file_name + "\n", log_file_name)
            try:
                client.close()
            except:
                pass
            tries -= 1
            if tries == 0:
                c = True
    if c:
        os.remove(absolute_file)
        return

    temp_file = os.path.join(dir_name, file_name.replace('.txt', '.temp.txt'))
    lang_temp_file = os.path.join(dir_name, file_name.replace('.txt', '.temp.lang.txt'))
    lang_file = os.path.join(dir_name, file_name.replace('.txt', '.lang.txt'))

    print "Converting " + absolute_file
    convert_file(absolute_file, temp_file)

    print "Detecting languages"
    subprocess.call("ldig/ldig.py -m ldig/models/model.latin " + temp_file + " >> " + lang_temp_file, shell=True)

    print "Post processing"
    process_lang_file(lang_temp_file, lang_file)

    print "Putting lang file on remote"
    tries = 2
    client = None
    while tries > 0:
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.load_system_host_keys()
            client.connect('157.193.215.41', username='cedric', password='v6SaMhg8SDxvxUfL')
            scp_client = scp.SCPClient(client.get_transport())
            scp_client.put(lang_file, dir_name)
            scp_client.close()
            client.close()
            tries = 0
        except:
            print "SCP Exception while putting file " + file_name
            write_log("SCP Exception while putting file " + file_name + "\n", log_file_name)
            try:
                client.close()
            except:
                pass
            tries -= 1

    print "Clean up"
    os.remove(temp_file)
    os.remove(lang_temp_file)
    os.remove(lang_file)
    os.remove(absolute_file)

def give_out_err(chan_out, chan_err):
    out = ""
    err = ""
    for line in chan_out:
        out += line
    for line in chan_err:
        err += line
    return out, err

def write_log(message, log_file_name):
    logs = open(log_file_name, 'wa')
    logs.write(message)
    logs.close()


if __name__ == '__main__':
    dir_name = sys.argv[1]
    convert_dir(dir_name)