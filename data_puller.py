# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import tags
from converter import Converter
import os


class DataGetter():
    def __init__(self, number_of_tags, number_of_pages, url, html_el, ids, tags, page_or_not=False, save='saver.txt'):
        self.number_of_tags = number_of_tags
        self.number_of_pages = number_of_pages
        self.web_url = url
        self.html_el = html_el
        self.id = ids
        self.page_or_not = page_or_not
        self.tagger = tags
        self.save = save

        self.use_page = False
        self.page = 1

        if self.save == 'saver.txt':
            self.converter = Converter()
        else:
            self.converter = Converter(save=self.save)
        
        self.comments = []

        self.tag = ""
        self.tag_counter = 1

        if self.page_or_not is True:
            self.url = self.web_url + self.tag + "/" + str(self.page)
        else:
            self.url = self.web_url + self.tag

        self.headers_param = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36 OPR/74.0.3911.160"}
        try:
            self.request = self.request = requests.get(self.url, headers=self.headers_param)
            self.soup = BeautifulSoup(self.request.content, "lxml")

        except requests.exceptions.ConnectionError:
            print("connection refused")
            quit()

    def update_url(self):
        if self.tagger == 1:
            self.tag = tags.tags1[self.tag_counter // self.nb_of_page]
        
        else:
            self.tag = tags.tags2[self.tag_counter // self.nb_of_page]
        
        if self.page_or_not is True:
            self.url = self.web_url + self.tag + "/" + str(self.page)
        
        else:
            self.url = self.web_url + self.tag

        print(self.url)

    def get_comments(self):
        if self.html_el is False:
            self.finder = self.soup.find_all(self.html_el)
        else:
            self.finder = self.soup.find_all(self.html_el, self.id)

        if not self.finder:  # <== check for NoneType
            print('element not found')
            self.finder = 'no text'
            self.use_page = False

        else:
            for link in self.finder:  # <== Get all comments from a page with for loop
                likn_text = link.text
                likn_text.strip()

                print(str(likn_text))
                self.comments.append(likn_text)
                self.converter.save_txt(likn_text)
                print("--------------------------------------------------------------------------------------------")
            
            del self.comments[:]

            self.use_page = True
            print(self.url)

        if self.use_page is True:
            self.page += 1
            self.url = self.web_url + self.tag + "/" + str(self.page)
            if self.page > self.nb_of_page:
                self.page = 1
                self.use_page = False

        print("======================")
        print(self.request.status_code)

    def get_data(self):
        for i in range(self.nb_of_tags):  # <=== getting tags from tags
            if self.use_page is not True:
                self.update_url()  # <=== update the url

            self.request = requests.get(self.url, headers=self.headers_param)
            self.soup = BeautifulSoup(self.request.content, "lxml")
            self.get_comments()

            print(self.tag_counter)

            i += 1
            self.tag_counter += 1
        

#datas = DataGetter(308, 7, "https://www.uludagsozluk.com/k/", 'div', {"class": "entry-p"}, tagger=1, page_or_not=True, save='new_data.txt')
#data = DataGetter(143, 1, "https://tr.wikipedia.org/wiki/", 'p', ids=False, tagger=2, page_or_not=False, save='text_file_tr.txt')
#143
