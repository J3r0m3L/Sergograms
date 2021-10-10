from scanner import main, toggleon, toggleoff
import eel 

  
eel.init('GUI')
eel.start('index.html', block=False)

@eel.expose
def toggle(boolean):
    if boolean == 'true':
        toggleon()
        print('true_good')
    else:
        toggleoff()
        print('false_good')

main()



