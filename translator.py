import time
from selenium import webdriver
import selenium 

class Sel_Translator:
    def __init__(self, headers_param={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36 OPR/74.0.3911.160"},
        url="https://translate.google.com.tr/?sl=tr&tl=en&text=",
        url_pretag= '%20',
        read_data='my_data.txt',
        write_data='your_data_n_target_lang.txt',
        sleep=2):

        self.read_data = open(read_data, "r", encoding='UTF-8')
        self.write_data = open(write_data, "a", encoding="UTF-8")

        self.sleep = sleep

        self.url = url
        self.pure_url = url #<--- we dont change anything on this string

        self.headers_param = headers_param
        self.url_pretag  = url_pretag

        self.words_array = []
        self.url_words_array = []
        self.urls = []
        comments = []
        ssusuz_array = []

        self.counter = 0
    
    def create_url(self, data):
        for line in data:  #<--- Splitting the lines to craete urls of sentences
            words = line.split()
            self.words_array.append(words)

        for splitted_sentences in self.words_array:  
            url_words = ''
            for each_sentences_words in splitted_sentences:  #<--- multi for loop takes the words from the splitted sentences array. self.words_array's inside => [['I', 'like', 'you'], ['lets', 'make', 'it']]
                url_words = url_words + str("%20"+ each_sentences_words)
            self.url_words_array.append(url_words) #<--- new array is like => [['%20I', '%20like', '%20you'], ['%20lets', '%20make', '%20it']]

        while True:
            try:
                self.url_words_array.remove("") # <-- Removing unnecessary elements from the array
            except ValueError:
                break

        translate_url = ''
        for url_word in self.url_words_array:
            self.url = self.pure_url
            translate_url = self.url + url_word
            self.urls.append(translate_url)
    
    def translate_and_save(self):
        self.browser = webdriver.Chrome()
        self.create_url(self.read_data)
        for i in self.urls:
            try:   
                self.browser.get(i)
                time.sleep(self.sleep)

                l = self.browser.find_element_by_xpath('//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[2]/c-wiz[2]/div[5]/div/div[1]/span[1]/span/span')
                a = self.browser.find_element_by_xpath('//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[2]/c-wiz[1]/span/span/div/textarea')

                print('text is: ' + l.text)

                second = l.text + '.'
                first = a.text + '.'

                second.replace('\n', '').replace('\t', '').replace('\r', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '')
                first.replace('\n', '').replace('\t', '').replace('\r', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '')

                self.write_data.writelines(first + '\t' + second + '\n')

                self.counter += 1
                print(self.counter)
            
            except Exception as error:
                print(error)
                pass

        self.browser.close()


sa = Sel_Translator()
sa.translate_and_save()