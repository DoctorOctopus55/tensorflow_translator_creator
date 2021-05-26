from self_translate import TModel
from data_puller import DataGetter
from translator import Sel_Translator

translator = Sel_Translator()
data_pull = DataGetter(143, 1, "https://tr.wikipedia.org/wiki/", 'p', ids=False, tags=2, page_or_not=False, save='my_data.txt')
model = TModel

def translator_():
    translator.translate_and_save()

def data_pull():
    data.get_data()

def model_train():
    model.train(self, epoch=10, batchsize=256)

def talk():
    model.talk()

input_for_what = input('What do you want me to do?')

if input_for_what == 'run_translator':
    translator_()

if input_for_what == 'data_pull':
    data_pull()

if input_for_what == 'train':
    model_train()

if input_for_what == 'talk':
    talk()