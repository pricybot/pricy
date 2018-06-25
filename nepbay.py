import json

with open("nepbay.json", "r") as read_file:
    data2 = json.load(read_file)
#print(data2[1]['Model'])   


def nepbay_data():
    nepbay_model = []
    for i in range(38):
        nepbay_model.append(data2[i]['Model'])
    #list of available models in daraz
    print(nepbay_model)

def nepbay_price(usermodel):
    usermodel = usermodel.upper()
    nepbay_model = []
    for i in range(38):
        nepbay_model.append(data2[i]['Model'])
    if usermodel in nepbay_model:
        #index of usermodel
        x = nepbay_model.index(usermodel)
        extract_model = data2[x]['Model']
        if extract_model == usermodel:
            print("Pricy: In "+ data2[x]['Website'] +" , price of "+ data2[x]['Brand'], data2[x]['Model'] +" is Rs. "+data2[x]['Price'])
            print("Pricy: Link "+ data2[x]['Link'])
            path = './'
            fileName = 'compare'
            dataset = {}
            dataset['Website'] = data2[x]['Website']
            dataset['Brand'] = data2[x]['Brand']
            dataset['Model'] = data2[x]['Model']
            dataset['Price'] = data2[x]['Price']
            dataset['Link'] = data2[x]['Link']

            def writeToJSONFile(path, fileName, dataset):
                filePathNameWExt = './' + path + '/' + fileName + '.json'
                with open(filePathNameWExt, 'a') as fp:
        
                    json.dump(dataset, fp, indent=4)

            writeToJSONFile(path, fileName, dataset)



#nepbay_price("j7")


def nepbay_compare(usermodel):
    usermodel = usermodel.upper()
    nepbay_model = []
    for i in range(38):
        nepbay_model.append(data2[i]['Model'])
    if usermodel in nepbay_model:
        #index of usermodel
        x = nepbay_model.index(usermodel)
        extract_price = data2[x]['Price']
        return extract_price

#nepbay_compare("j7")
