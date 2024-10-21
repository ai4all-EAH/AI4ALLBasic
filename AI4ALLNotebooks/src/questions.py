# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:13:49 2022

@author: class

part of the JuPy Project
2020

Original Version of the create_multipleChoice_widget() seems to be from: 
    
 https://github.com/dingandrew/Zumi-Python-Lessons/blob/master/Quiz_Generator.py   
"""



import numpy as np
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
from IPython.display import YouTubeVideo
from IPython.display import Markdown
from IPython.display import Code


def create_multipleChoice_widget(description, options, correct_answer,last=0):
    if correct_answer not in options:
        options.append(correct_answer)
    
    correct_answer_index = options.index(correct_answer)
    
    radio_options = [(words, i) for i, words in enumerate(options)]
    alternativ = widgets.RadioButtons(
        options = radio_options,
        description = '',
        disabled = False
    )
    
    description_out = widgets.Output()
    with description_out:
        display(description)
        
    feedback_out = widgets.Output()

    def check_selection(b):
        a = int(alternativ.value)
        if a==correct_answer_index:
            s = '\x1b[6;30;42m' + "Richtig." + '\x1b[0m' +"\n" #green color
        else:
            s = '\x1b[5;30;41m' + "Falsch. " + '\x1b[0m' +"\n" #red color
        with feedback_out:
            clear_output()
            print(s)
        return
    
    check = widgets.Button(description="Ã¼berprÃ¼fen")
    check.on_click(check_selection)
    
    
    return widgets.VBox([description_out, alternativ, check, feedback_out])



def create_multipleChoice_widget_new(descriptionText, descriptionCode, options, correct_answer, last=0):
    if correct_answer not in options:
        options.append(correct_answer)
    
    correct_answer_index = options.index(correct_answer)
    
    radio_options = [(words, i) for i, words in enumerate(options)]
    alternativ = widgets.RadioButtons(
        options = radio_options,
        description = '',
        disabled = False,
        layout=widgets.Layout(width='auto')
    )
    
    description_out = widgets.Output()
    with description_out:
        if len(descriptionText)>0:
                m=Markdown(descriptionText)
                display(m)
        if len(descriptionCode)>0:
                #c=Code(descriptionCode,language='python')
                c=Markdown("```python\n"+descriptionCode+"\n```")
                display(c)
        
        
    feedback_out = widgets.Output()

    def check_selection(b):
        a = int(alternativ.value)
        if a==correct_answer_index:
            s = '\x1b[6;30;42m' + "Richtig." + '\x1b[0m' +"\n" #green color
        else:
            s = '\x1b[5;30;41m' + "Falsch. " + '\x1b[0m' +"\n" #red color
        with feedback_out:
            clear_output()
            print(s)
        return
    
    check = widgets.Button(description="Ã¼berprÃ¼fen")
    check.on_click(check_selection)
    
    
    return widgets.VBox([description_out, alternativ, check, feedback_out])



def create_int_input_widget(descriptionText, descriptionCode, correct_answer, last=0):

    
    description_out = widgets.Output()
    with description_out:
        if len(descriptionText)>0:
                m=Markdown(descriptionText)
                display(m)
        if len(descriptionCode)>0:
                #c=Code(descriptionCode,language='python')
                c=Markdown("```python\n"+descriptionCode+"\n```")
                display(c)
        
    eingabe = widgets.IntText(
        description='Wert:',
        disabled=False
    )
    
    
    
    check = widgets.Button(description="Ã¼berprÃ¼fen")
    
    
    feedback_out = widgets.Output()

    def check_selection(b):
        given_answer = int(eingabe.value)
        with feedback_out:
            clear_output()
            if given_answer==correct_answer:
                print('\x1b[6;30;42m' + "Richtig!" + '\x1b[0m' +"\n") #green color
            elif given_answer!=correct_answer:
                print('\x1b[5;30;41m' + "Falsch!" + '\x1b[0m' +"\n") #red color
        return

    check.on_click(check_selection)
    
    return widgets.VBox([description_out, eingabe, check, feedback_out])




# allows for specific feedback in case of a wrong value
# feedback_dic is a dictionary,
# teh wrong answer is the key, the feedback (of type string) the value 
def create_int_input_widget_feedback(descriptionText, descriptionCode, correct_answer, feedback_dic, last=0):

    
    description_out = widgets.Output()
    with description_out:
        if len(descriptionText)>0:
                m=Markdown(descriptionText)
                display(m)
        if len(descriptionCode)>0:
                #c=Code(descriptionCode,language='python')
                c=Markdown("```python\n"+descriptionCode+"\n```")
                display(c)
        
    eingabe = widgets.IntText(
        description='Wert:',
        disabled=False
    )
    
    
    feedback_out = widgets.Output()

    def check_selection(b):
        given_answer = int(eingabe.value)
        if given_answer==correct_answer:
            s = '\x1b[6;30;42m' + "Richtig!" + '\x1b[0m' +"\n" #green color
        elif given_answer in feedback_dic.keys():
            s = '\x1b[5;30;41m Falsch!'+feedback_dic[given_answer] + '\x1b[0m' +"\n" #red color     
        else:
            s='\x1b[5;30;41m' + "Falsch!" + '\x1b[0m' +"\n" #red color   
        with feedback_out:
            clear_output()
            print(s)
        return
    
    check = widgets.Button(description="Ã¼berprÃ¼fen")
    check.on_click(check_selection)
    
    
    return widgets.VBox([description_out, eingabe, check, feedback_out])