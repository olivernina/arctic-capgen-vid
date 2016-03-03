
from bs4 import BeautifulSoup as Soup, SoupStrainer
import urllib
import os


def video(video_dir,video_name,video_clip):


    url='http://courvila_contact:59db938f6d@lisaweb.iro.umontreal.ca/transfert/lisa/users/courvila/data/lisatmp2/torabi/DVDtranscription/'+video_name+'/video/'+video_clip
    # u1 = urllib.urlopen(url)
    # soup = Soup(u1)

    u2 = urllib.urlopen(url)
    video_dir_dst = os.path.join(video_dir,video_name)
    if not os.path.exists(video_dir_dst):
        os.mkdir(video_dir_dst)

    f = open(video_dir_dst+'/'+video_clip, 'wb')
    meta = u2.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (video_name, file_size)

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u2.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,

    f.close()


def video_mpii(video_dir,video_name,video_clip):


    # url='http://courvila_contact:59db938f6d@lisaweb.iro.umontreal.ca/transfert/lisa/users/courvila/data/lisatmp2/torabi/DVDtranscription/'+video_name+'/video/'+video_clip
    url='http://97H5:thoNohyee7@datasets.d2.mpi-inf.mpg.de/movieDescription/protected/avi/'+video_name+'/'+video_clip



    u2 = urllib.urlopen(url)
    video_dir_dst = os.path.join(video_dir,video_name)
    if not os.path.exists(video_dir_dst):
        os.mkdir(video_dir_dst)

    f = open(video_dir_dst+'/'+video_clip, 'wb')
    meta = u2.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (video_name, file_size)

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u2.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,

    f.close()


def main():
    video_dir = "TITANIC_1"
    url='http://courvila_contact:59db938f6d@lisaweb.iro.umontreal.ca/transfert/lisa/users/courvila/data/lisatmp2/torabi/DVDtranscription/'+video_dir+'/video/'
    u1 = urllib.urlopen(url)
    soup = Soup(u1)

    links = soup.find_all('a')

    for link in links:
        if link.has_attr('href') and link['href'].find('.avi')>=0:
            video_name  =  link['href']

            u2 = urllib.urlopen(url+video_name)
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)

            f = open(video_dir+'/'+video_name, 'wb')
            meta = u2.info()
            file_size = int(meta.getheaders("Content-Length")[0])
            print "Downloading: %s Bytes: %s" % (video_name, file_size)

            file_size_dl = 0
            block_sz = 8192
            while True:
                buffer = u2.read(block_sz)
                if not buffer:
                    break

                file_size_dl += len(buffer)
                f.write(buffer)
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
                status = status + chr(8)*(len(status)+1)
                print status,

            f.close()
