from extract_txt import read_files
from txt_processing import preprocess
from txt_to_features import txt_features, feats_reduce
from extract_entities import get_number, get_email, rm_email, rm_number, get_name, get_skills, get_location
from model import simil
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')


if __name__=="__main__":
    directory = '/Users/development/Documents/GitHub/Resume-Parser-Shortlisting-Project/Data'
    resume_path = '/Users/development/Documents/GitHub/Resume-Parser-Shortlisting-Project/files/resumes'
    jd_path = '/Users/development/Documents/GitHub/Resume-Parser-Shortlisting-Project/Data/JobDesc'
    resumetxt=read_files(resume_path)
    p_resumetxt = preprocess(resumetxt)

    jdtxt=read_files(jd_path)
    p_jdtxt = preprocess(jdtxt)
    
    feats = txt_features(p_resumetxt, p_jdtxt)
    feats_red = feats_reduce(feats)

    df = simil(feats_red, p_resumetxt, p_jdtxt)

    t = pd.DataFrame({'Original Resume':resumetxt})
    dt = pd.concat([df,t],axis=1)
    dt['Phone No.']=dt['Original Resume'].apply(lambda x: get_number(x))
    
    dt['E-Mail ID']=dt['Original Resume'].apply(lambda x: get_email(x))

    dt['Original']=dt['Original Resume'].apply(lambda x: rm_number(x))
    dt['Original']=dt['Original'].apply(lambda x: rm_email(x))
    dt['Candidate\'s Name']=dt['Original'].apply(lambda x: get_name(x))

    skills = pd.read_csv('/Users/development/Documents/GitHub/Resume-Parser-Shortlisting-Project/Data/skill_red.csv')
    skills = skills.values.flatten().tolist()
    skill = []
    for z in skills:
        r = z.lower()
        skill.append(r)



    dt['Skills']=dt['Original'].apply(lambda x: get_skills(x,skill))
    print(dt['Location'].head(30))

    # Load the database of job descriptions and required skills
    job_db_path = '/Users/development/Documents/GitHub/Resume-Parser-Shortlisting-Project/Data/job_db.csv'
    job_db = pd.read_csv(job_db_path)

    # Function to calculate accuracy
    def calculate_accuracy(extracted_skills, required_skills):
        matched_skills = set(extracted_skills).intersection(set(required_skills))
        accuracy = len(matched_skills) / len(required_skills) if required_skills else 0
        return accuracy

    # Compare extracted skills with job descriptions in the database
    accuracy_list = []
    for index, row in dt.iterrows():
        job_desc = row['Original Resume']
        extracted_skills = row['Skills']
        
        # Find the matching job description in the database
        matching_job = job_db[job_db['Job Description'] == job_desc]
        if not matching_job.empty:
            required_skills = matching_job['Skills Required'].values[0].split(',')
            accuracy = calculate_accuracy(extracted_skills, required_skills)
            accuracy_list.append(accuracy)
        else:
            accuracy_list.append(0)

    # Add accuracy to the dataframe
    dt['Accuracy'] = accuracy_list

    # Print the dataframe with accuracy
    print(dt[['Original Resume', 'Skills', 'Accuracy']].head(30))

    # Save the dataframe to a CSV file

    dt.to_csv('/Users/development/Documents/GitHub/Resume-Parser-Shortlisting-Project/Data/output.csv', index=False)


