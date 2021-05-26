import io
import time

class Converter():
    def __init__(self, save="text_file_tr.txt"):
        self.comments = open(save, "a", encoding="utf-8")

        self.ranges = 1

        self.rows = None
        self.value = None
    
    def fix_string(self, string):
        fixstring = string.replace('\n', "  ").replace("\t", "  ").replace("[", "  ").replace("]", "  ").replace("(", "").replace(")", "")
        string_array = fixstring.split('.')

        if string.startswith('www') is True:
            fixstring = fixstring.replace(string, '')
        if string.startswith('(bknz:') is True:
            fixstring = fixstring.replace(string, '')

        return string_array

    def save_txt(self, comment):
        self.sent_array = self.fix_string(str(comment))    
        for i in range(len(self.sent_array)):
            sentence = str(self.sent_array[i])
            sentence.strip()
            self.comments.write((sentence) + '\n')

