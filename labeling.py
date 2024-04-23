import os
import pandas as pd

os.chdir('./images')

folder_names = os.listdir()

for folder_name in folder_names:
    print(os.path.splitext(folder_name)[0])

names=[]

for f in folder_names:
    os.chdir('./%s' % f)
    file_names = os.listdir()
    for file_name in file_names:
        name = f+'/'+os.path.splitext(file_name)[0]+'.jpg'
        names.append(name)         ## 모든 파일명 리스트화
    os.chdir('./..')    ## 다시 상위 폴더로

df = pd.DataFrame(names, columns=['file_name'])

hoodies = []
tshirt = []
longsleeve = []
sleeveless = []
croptop = []

for i in range(len(names)):
    check = df['file_name'][i]
    
    if check.startswith('sleeve'):
        sleeveless.append(1)
    else:
        sleeveless.append(-1)
    
    if check.startswith('long'):
        longsleeve.append(1)
    else:
        longsleeve.append(-1)

    if 'tshirt' in check:
        tshirt.append(1)
    else:
        tshirt.append(-1)

    if 'croptop' in check:
        croptop.append(1)
    else:
        croptop.append(-1)

    if 'hoodies' in check:
        hoodies.append(1)
    else:
        hoodies.append(-1)


hood= pd.DataFrame(hoodies, columns=['hoodie'])
long= pd.DataFrame(longsleeve, columns=['long_sleeve'])
t = pd.DataFrame(tshirt,columns=['t-shirt'])
less = pd.DataFrame(sleeveless,columns=['sleeveless'])
crop = pd.DataFrame(croptop,columns=['crop_top'])

target = pd.concat([df, hood, long, t, less, crop], axis=1)
print(target)


target.to_csv('list_attr_celeba.txt', sep=' ', index=False)
