# given Img_num
# Randomize data, split into 5 folds


import random	
	


# define number of images
Img_num = 100

# create index
def Ind_list():
    Ind = range(Img_num)
    return Ind

def Randomized_Ind():
    Randomized = random.sample(Ind_list(),Img_num)
    return Randomized



#print Ind_list()
#print Randomized_Ind()
# print randomized list
#print random.sample(Ind_list(),Img_num)



# split data into 5 pieces

if Img_num%5==0:

    def Folds():
    	indices = Randomized_Ind()
        first = indices[0:(Img_num/5)]
        second = indices[(Img_num/5):2*(Img_num/5)]
        third = indices[2*(Img_num/5):3*(Img_num/5)]
        fourth = indices[3*(Img_num/5):4*(Img_num/5)]
        fifth = indices[4*(Img_num/5):5*(Img_num/5)]
        return first, second, third, fourth, fifth
        
    first_fold, second_fold, third_fold, fourth_fold, fifth_fold = Folds()

print first_fold





