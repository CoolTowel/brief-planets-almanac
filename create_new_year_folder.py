import datetime
import os

year = str(datetime.date.today().year+1)
os.makedirs('./data/'+year+'/40')
os.makedirs('./data/'+year+'/24')
os.makedirs('./data/'+year+'/52')

os.makedirs('./output/'+year+'/40')
os.makedirs('./output/'+year+'/24')
os.makedirs('./output/'+year+'/52')

