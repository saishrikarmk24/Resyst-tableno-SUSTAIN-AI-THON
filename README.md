# RESYST AQUAREGIA SUSTAIN-AI-THON
##Introduction:
Our Project is aimed at pursuing SDG 6 'Clean Water and Sanitation' and specifcally at section 6.3 which aims to reduce water pollution. Our solution consists of training a Machine Learning model on existing datasets sourced from Google Earth and Nasa EOSDIS to classify using a classification algorithm the type of water-pollutant it is. (small, medium or large). We also allow users to interact with and search for a water body near them. 

##Workflow:
1. Data Collection
Satellite Data: Google Earth Engine and NASA EOSDIS.
Focus on multispectral bands for water quality analysis.
Historical data for pattern recognition

2. Image Preprocessing
Download satellite images.
Cloud masking and atmospheric correction.
Preprocess (resize, normalize and augment) for machine learning analysis.

3. Machine Learning Model Development
Model Selection: Object detection (YOLO and Mask R-CNN).
Model Training: Label data, split into training/validation sets, and train the model.
Spectral analysis for pollution type classification
Evaluation: Measure accuracy, precision, and recall to fine-tune the model.

4. Garbage Detection & Pollution Estimation
Detect garbage in satellite images using the trained model.
Estimate pollution levels based on detected garbage size.

5.  User Interface & Accessibility

Web Application:
Interactive dashboard built with React.js
Pollution hotspots displayed on interactive maps
Historical trend graphs.
Community reporting integration

Key Features:
Area-wise pollution level indicators
Search by location/water body name
Mobile-responsive design

User Access:
Role-based authentication
Public view for basic pollution maps
Advanced analytics for premium users


##Concept Map:











##Tech Stack:
Frontend: HTML, CSS, Javascript (React.js)
Backend: Flask
Database: SQLite, SQLAlchemy
ML Model: Pytorch, MatPlotLib, numpy

##Novelty:
Our USP (Unique Selling Proposition) is a satellite-based pollution detection system using Artificial Intelligence and Machine Learning models that provides comprehensive pollution mapping without hardware dependencies. 

Novelty:
Publicly accessible pollution mapping
Scalable without hardware dependencies
Historical pollution pattern analysis

We plan to make our product scalable by integrating more advanced AI models, expanding datasets, and working with more NGOs and corporations for a more comprehensive adoption worldwide.

##Solution: 
Our solution utilises satellite data from Google Earth Engine and NASA EOSDIS, focusing on multispectral bands for water quality analysis and historical patterns. Images undergo cloud masking, atmospheric correction, and preprocessing for machine learning. Using YOLO and Mask R-CNN, we train models for object detection and spectral pollution classification. An interactive React.js dashboard displays pollution hotspots, historical trends, and location-based searches. With role-based authentication, the platform offers public pollution maps and advanced analytics, empowering users to monitor and mitigate water pollution effectively.

#Others:
##Business Model
Our project envisions a SaaS (Software as a Service) business model with a tiered structure. The free tier provides access to basic pollution maps. Premium users gain access to historical analysis and prediction models, while enterprise clients receive custom analysis capabilities, API access, and dedicated support. Target customers include environmental agencies, municipal corporations, research institutions, and environmental NGOs.

##Aligning with SDG 6:
This project directly aligns with UN Sustainable Development Goal 6 (Clean Water and Sanitation), specifically target 6.3 which aims to improve water quality by reducing pollution and minimizing the release of hazardous materials into water bodies. By using satellite images and Artificial Intelligence for pollution detection, the product enables data-driven decision making and timely interventions to protect water resources, ultimately helping improve water quality and the ecosystem.



 

