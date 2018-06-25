import json

#importing daraz.json file
with open("daraz.json", "r") as read_file:
    data1 = json.load(read_file)
#print(data1[1]['Model'])
    


def daraz_data():
    daraz_model = []
    for i in range(36):
        daraz_model.append(data1[i]['Model'])
    #list of available models in daraz
    print(daraz_model)

def daraz_price(usermodel):
    usermodel = usermodel.upper()
    daraz_model = []
    for i in range(36):
        daraz_model.append(data1[i]['Model'])
    if usermodel in daraz_model:
        #index of usermodel
        x = daraz_model.index(usermodel)
        extract_model = data1[x]['Model']
        if extract_model == usermodel:
            print("Pricy: In "+ data1[x]['Website'] +" , price of "+ data1[x]['Brand'], data1[x]['Model'] +" is Rs. "+data1[x]['Price'])
            print("Pricy: Link "+ data1[x]['Link'])
         
            
#daraz_price("j7")

def daraz_compare(usermodel):
    usermodel = usermodel.upper()
    daraz_model = []
    for i in range(36):
        daraz_model.append(data1[i]['Model'])
    if usermodel in daraz_model:
        #index of usermodel
        x = daraz_model.index(usermodel)
        extract_price = data1[x]['Price']
        return extract_price

#daraz_compare("j7")
