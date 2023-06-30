import time

def file_split():
    lines_per_file = 100000
    smallfile = None
    i=0
    with open('news.tokenized.shuffled.txt') as bigfile:  #news.tokenized.shuffled.txt
        for lineno, line in enumerate(bigfile):
            if lineno % lines_per_file == 0:
                if smallfile:
                    smallfile.close()
                small_filename = '/Applications/CS_diploma_code/wordrep/small_corpus_{}.txt'.format(i)
                i=i+1
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()

def main():
    file_split()




if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')