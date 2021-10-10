from scanner import main, toggleon, toggleoff
import eel 

  
eel.init('GUI')
eel.start('index.html', block=False)

main()
@eel.expose
def toggle(boolean):
    if boolean == true:
        toggleon()
        print('true_good')
        eel.showHistory('true works')
    else:
        toggleoff()
        print('false_good')
        eel.showHistory('false works')





