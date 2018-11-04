import warnings 
warnings.filterwarnings(action = 'ignore') 

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pickle
import time
import gensim
import re


###########   HELPER FUNCTIONS #############

def remove_punc(inp):
    #s = re.sub(r'[^\w\s]','',s)
    inp=inp.replace(';',' ')
    return re.sub(r'[^\w\s]','',str(inp))
def alpha_order(inp):
    return ' '.join(sorted(inp.split(',')))

def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))
def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def ascii_rep(x):
    lst=np.array([ord(ch) for ch in x])
    return ((lst/ord('z')).sum())/len(lst)

def word2vec_feat(lst_words,model):
    fin_vec=[]
    for wrd in lst_words:
     try:
        fin_vec.append(model[wrd])
     except:
         fin_vec.append(np.zeros(150).tolist())
    fin_vec=np.array(fin_vec)
    wd_vec_feat=fin_vec.sum(axis=0)
    return wd_vec_feat
 

#-- NORMALIZES THE VECTOR --#
def normalize_vect(lst):
    return lst/np.sqrt((lst**2).sum())  #v/np.sqrt((v**2).sum())

def common_feat(sentences):
    vectorizer=TfidfVectorizer()
    tfidf=vectorizer.fit_transform(sentences)
    cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
    return cosine_similarities

file_to_read='test.xlsx'
pkl_to_save='feat_extracted_stp_1.pkl'
fin_file_store='fin_df1.pkl'


#df =pd.read_pickle("data_file_pd.pkl")
def feature_extract(df):
   # print('---------------------------------------------------------------------')
    print('---------------BASE FEATURES EXTRACTION------------------------------')
    df['productFamily']=df['productFamily'].apply(alpha_order)
    df['train_set'] = df[['title','description','productBrand','productFamily','size','color','keySpecsStr','sellerName']].apply(lambda x: ' '.join(map(str, x)), axis=1)
    df['Removed_punc']=df['train_set'].apply(remove_punc)
    df['len_row']= df['Removed_punc'].apply(lambda x: len(str(x)))
    print('Length of rows DONE')
    df['len_char']=df['Removed_punc'].apply(lambda x : len(''.join(set(str(x).replace(' ','')))))
    print('Length of characters DONE')
    df['len_word']=df['Removed_punc'].apply(lambda x :len(str(x).split()))
    print('Length of words DONE')
    df['avg_word']=df['Removed_punc'].apply(lambda x: avg_word(x))
    print('Average of words DONE')
    df['numerics'] = df['Removed_punc'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
    print('Check Numerics DONE')
    df['upper_count'] = df['Removed_punc'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
    print('Upper case counts DONE')
    df['lower_case']=df['Removed_punc'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    print('Check Lowercases DONE')
    #freqH= pd.Series(' '.join(df['lower_case']).split()).value_counts()[:10]
    #df['rmv_most_occuring_words'] = df['lower_case'].apply(lambda x: " ".join(x for x in x.split() if x not in freqH))  
    stop = stopwords.words('english')
    df['rmv_stop_word'] = df['lower_case'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    print('Remove stop words DONE')
    df['pre_process']=df['rmv_stop_word'].apply(lambda x : gensim.utils.simple_preprocess(x))
    print('Preprocessing DONE')
    return df

############################################################################################
#EXTRACTED FEATURE DF



#BUILD WORD TO VEC MODEL
#model = gensim.models.Word2Vec(
#       new_df['pre_process'],
#        size=150,
#        window=10,
#        min_count=2,
#        workers=10)
#model.train(new_df['pre_process'], total_examples=len(new_df['pre_process']), epochs=10)
#model.save('model.bin')
##############################################################################################

#READ SAVED MODELS AND FILES
#new_df = pd.read_pickle(pkl_to_save)
#model = gensim.models.Word2Vec.load('model.bin')
#print(new_df['pre_process'][0])
###############################################################################################
#TF-IDF VECTORIZER
#print(new_df['len_row'][0])
#def combine_vect(x):
#    lst_feature=[]
#    lst_feature.append(x['len_row'])
#    lst_feature.append(x['len_char'])
#    lst_feature.append(x['len_word'])
#    lst_feature.append(x['avg_word'])
#    lst_feature.append(x['numerics'])
#    lst_feature.append(x['upper_count'])
#    lst_feature.append(word2vec_feat(x['pre_process']))
#    return lst_feature


#for i in range(5):
    


def assemble_features(new_df,model):
   
    print('-----------------ASSEMBLING FEATURES STARTED-------------------------')
    fin_df = pd.DataFrame(columns=['lst_vect','rmv_word'])
    percent=10
    for i in range(new_df.shape[0]):
        lst_feature=[]
        lst_feature.append(new_df['len_row'][i])
        lst_feature.append(new_df['len_char'][i])
        lst_feature.append(new_df['len_word'][i])
        lst_feature.append(new_df['avg_word'][i])
        lst_feature.append(new_df['numerics'][i])
        lst_feature.append(new_df['upper_count'][i])
        lst_feature+=word2vec_feat(new_df['pre_process'][i],model).tolist()
        fin_df.loc[i] = [lst_feature,new_df['rmv_stop_word'][i]]
        if int(i%(new_df.shape[0]*10/100))==0:
            print(percent,'%','completed..')
            percent+=10
    print('100%','completed..')
    print('---------------------------------------------------------------------')
    return fin_df


def complete_feature(inp_df):
    print('------------------COMPLETE FEATURE EXTRACTION STARTED----------------')
    data_df = pd.DataFrame(columns=['features'])
    percent=10
    for indx in range(inp_df.shape[0]):
        lst_vect1=normalize_vect(np.array(inp_df['lst_vect'][indx])).tolist()
        #print(lst_vect1)
        lst_vect2=normalize_vect(np.array(inp_df['lst_vect2'][indx])).tolist()
        #print(lst_vect2)
        #print(inp_df['rmv_word'][indx])
        #print(inp_df['rmv_word2'][indx])
        cosine_sim=common_feat([inp_df['rmv_word'][indx],inp_df['rmv_word2'][indx]]).tolist()
        #print(cosine_sim)
        sim_vect3=sum(cosine_sim)/len(cosine_sim)
        #print(sim_vect3)
        jac_vect4=get_jaccard_sim(inp_df['rmv_word'][indx],inp_df['rmv_word2'][indx])
        #print(jac_vect4)
        feat_vect=lst_vect1+lst_vect2
        feat_vect.append(sim_vect3)
        feat_vect.append(jac_vect4)
        #print(feat_vect)
        data_df.loc[indx] = [feat_vect]
        if int(indx%(len(inp_df)*10/100))==0:
            print(percent,'%','completed..')
            percent+=10
    print('100%','completed..')
    print('-------------------------------------------------------------------')
    feature_set=np.array([i for i in data_df['features']])
    return feature_set
      


    

#print(fin_df['lst_vect'][0])
#print(fin_df['rmv_word'][0])
#fin_df.to_pickle(fin_file_store)

#print(common_feat([new_df['rmv_stop_word'][0],new_df['rmv_stop_word'][0]]))
#print(get_jaccard_sim(new_df['rmv_stop_word'][0],new_df['rmv_stop_word'][0]))


def main(file_to_read,mode='test'):
    
    print('-------------------READING THE FILE--------------------------------')
    df = pd.read_excel(file_to_read,converters={'productId':str,'title':str,'description':str,'mrp':str,'sellingPrice':str,'specialPrice':str,'categories':str,'productBrand':str,'productFamily':str,'size':str,'color':str,'keySpecsStr':str,'sellerName':str})
    new_df=feature_extract(df)
    # SAVES AS PKL FILE WHICH WILL BE GREATLY USED IN CASE OF ANY UNCERTAINITIES
    new_df.to_pickle('make_features_df.pkl')
    model=None

    if(mode=='train'):
        #BUILD WORD TO VEC MODEL
        model = gensim.models.Word2Vec(
                new_df['pre_process'],
                size=150,
                window=10,
                min_count=2,
                workers=10)
        model.train(new_df['pre_process'], total_examples=len(new_df['pre_process']), epochs=10)
        model.save('model.bin')
        assembled_features_df=assemble_features(new_df,model)
        assembled_features_df.to_pickle('assembled_features_df.pkl')
        
        print('-----TWO FEATURE EXTRACTION FOR POSITIVE AND NEGATIVE SCENARIOS------')
        #BUILDING DATA FOR NON DUPLICATE DATA
        except_top_two_df = assembled_features_df.iloc[2:]
        top_two_df=except_top_two_df.head(2)

        two_stepped_down_df=except_top_two_df.append(top_two_df,ignore_index=True)
        
        temp_df=pd.DataFrame()
        temp_df['lst_vect2']=two_stepped_down_df['lst_vect']
        temp_df['rmv_word2']=two_stepped_down_df['rmv_word']

        no_dup_df=pd.concat([assembled_features_df,temp_df],axis=1)
        
        feat_pos=complete_feature(no_dup_df)

        #BUILDING DATA FOR DUPLICATE DATA
        df_duplicate=pd.DataFrame()
        df_duplicate['lst_vect2']=assembled_features_df['lst_vect']
        df_duplicate['rmv_word2']=assembled_features_df['rmv_word']
        dup_df=pd.concat([assembled_features_df,df_duplicate],axis=1)

        feat_neg=complete_feature(dup_df)
        
        X_train=np.concatenate((feat_pos,feat_neg),axis=0)
        y = np.hstack((np.zeros(len(feat_pos)), np.ones(len(feat_neg))))

        ## TRAINING THE MODEL  
        svc=LinearSVC()
        rand_state = np.random.randint(0, 1000)
        X_t, X_test, y_t, y_test = train_test_split(X_train, y, test_size=0.2, random_state=rand_state)
        t=time.time()
        svc.fit(X_t, y_t)
        t2 = time.time()
        fittingTime = round(t2 - t, 2)
        accuracy = round(svc.score(X_test, y_test),4)
        print('-------------------------------------------------------------------')
        print('Fitting Time : ',fittingTime)
        print('Accuracy : ',accuracy)
        print('-------------------------------------------------------------------')
        with open('svc_model.pkl', 'wb') as f:
            pickle.dump(svc, f)
        print('Model Saved as : svc_model.pkl')


    else:
        model = gensim.models.Word2Vec.load('model.bin')
        assembled_features_df=assemble_features(new_df,model)
        assembled_features_df=pd.concat([new_df['productId'],assembled_features_df],axis=1)
        temporary_df=assembled_features_df
        #temporary_df['duplicate_check_2']
        dict_dup_Id={}
        with open('svc_model.pkl', 'rb') as f:
            clf = pickle.load(f)
        
        for indx in range(len(assembled_features_df)):
            if(str(assembled_features_df['productId'][indx]) not in dict_dup_Id):
                 dict_dup_Id[str(assembled_features_df['productId'][indx])]=[]
        print('--------------------COMPARISION OF ROWS STARTED--------------------')
       
        for i in range(len(temporary_df)-1):
            print('Comparision Number: ',i)
            except_top_one_df =temporary_df.iloc[1:]
            except_top_one_df.index=[i for i in range(len(except_top_one_df))]
            temporary_df=except_top_one_df.append(temporary_df.head(1),ignore_index=True)
            df3=pd.DataFrame()
            df3['productId2']=temporary_df['productId']
            df3['lst_vect2']=temporary_df['lst_vect']
            df3['rmv_word2']=temporary_df['rmv_word']
            setup_df=pd.concat([assembled_features_df,df3],axis=1)
            feat_predict=complete_feature(setup_df)
            predictions=clf.predict(feat_predict)
            indx_lst=np.argwhere(predictions>0)
            for val in indx_lst:
                if(str(setup_df['productId'][val[0]]) in dict_dup_Id):
                    lst_val=dict_dup_Id[str(setup_df['productId'][val[0]])]
                    lst_val.append(str(setup_df['productId2'][val[0]]))
                    dict_dup_Id[str(setup_df['productId'][val[0]])]=lst_val
                    if(str(setup_df['productId2'][val[0]])) in dict_dup_Id:
                        del dict_dup_Id[str(setup_df['productId2'][val[0]])]
            #temporary_df=present_df
            #print(temporary_df)


if __name__=="__main__":
    main('machine_learn_data_xls.xlsx','train')