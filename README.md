### APTOS 2019 Blindness Detection
#### -Detect diabetic retinopathy to stop blindness before it's too late

#### What is diabetic retinopathy?
"Diabetic retinopathy is an eye condition that can cause vision loss and blindness in people who have diabetes. It affects blood vessels in the retina (the light-sensitive layer of tissue in the back of your eye).  

If you have diabetes, it’s important for you to get a comprehensive dilated eye exam at least once a year. Diabetic retinopathy may not have any symptoms at first — but finding it early can help you take steps to protect your vision." -https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy 

#### Real World Problem.
Currently trained technicians from Aravind Eye Hospital in India travel to rural areas where medical screening is difficult to conduct and capture images of retina, they then rely on hihgly trained doctors to review the images and provide diagnosis.
The goal is to scale their efforts through technology; to gain the ability to automatically screen images for disease and provide information on how severe the condition may be.

#### Mapping Real World Problem to a Machine Learning/Deep Learning Problem.
We are provided with data of thousands of images of Retina and we also have the corresponding diagnosis report for that image which tells us the siverity of diabetic retinopathy on a scale of 0-4.

This can be mapped to categorical classification problem, where our inputs will be images of retina and output will be a an integer between 0-4 where '0' means 'No DR' and '4' means 'Proliferative DR'.

#### Performance Metrics.
1. Confusion Matrix
2. Quadratic Weighted Kappa(QWK)
Kappa is a score which takes into account both accuracy of the model with respect to the doctor's diagnosis and also the agreement of the model and Doctor by chance it is represented by 'κ' and is defined as![image.png](attachment:image.png)
where po is the relative observed agreement among raters (identical to accuracy), and pe is the hypothetical probability of chance agreement, using the observed data to calculate the probabilities of each observer randomly seeing each category. If the raters are in complete agreement then kappa =1. If there is no agreement among the raters other than what would be expected by chance (as given by pe), kappa =0.

Weighted Kappa is a small variation to this, here if two raters disagree with each other then the score is given according to the distance of the ratings given by both raters.That means that our score will be higher if (a) the real value is 4 but the model predicts a as 3, and the score will be lower if (b) the model instead predicts a as 0.
