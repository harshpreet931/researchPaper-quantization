An IoT-Based Privacy-Preserving Federated Learning Framework for Proactive Machine Maintenance Prediction in Industry 5.0



Abstract
Purpose– This study aims to propose an IoT-based privacy-preserving Federated Learning Framework for predicting machine failures in Industry 5.0, addressing unexpected breakdowns that lead to production halts and increased operational costs.

Design/methodology/approach– The study employs an experimental research design using sensor data from industrial machines across multiple buildings. Each machine is equipped with IoT sensors measuring vibration, humidity, temperature, and pressure. The methodology compares ten AI models including Deep Learning approaches (MLP, LSTM, Autoencoders, CNN, Transformers, EfficientNet1D) and Federated Learning models (Channel-Separated CNN FL, Hierarchical FL, Adaptive FL, Ensemble FL), evaluated using accuracy, precision, recall, and F1-score metrics.

Findings– The research demonstrates that Federated Learning models consistently outperform Deep Learning approaches in predictive maintenance. Hierarchical FL achieved the highest accuracy of 98.44% with 99.24% precision, while EfficientNet1D showed the best recall at 87.58%. The results confirm that FL models provide data privacy through decentralized training without compromising predictive accuracy, making them ideal for distributed industrial environments.

Research limitations/implications– Some Deep Learning models exhibited high computational complexity, and FL constraints such as model divergence were observed. The findings enable industries to transition from reactive to predictive maintenance, reducing unplanned downtime and costs. The framework aligns with Industry 5.0 principles by letting AI handle monitoring while humans focus on complex decisions.

Originality/value– This research uniquely integrates Deep Learning and Federated Learning for Industry 5.0 predictive maintenance, addressing the often-overlooked aspects of data privacy and decentralized environments. The comparative analysis of ten models provides practitioners actionable insights for selecting appropriate solutions, while the framework's practical deployability using common sensors makes it readily applicable across manufacturing settings.

Keywords Industry 5.0, Internet of Things (IoT), Deep Learning, Federated Learning, Predictive maintenance, Machine downtime

Paper type Research paper

1. Introduction 

Industrial machinery plays a critical role in modern manufacturing and production processes, making reliability and efficiency paramount. The encountered problems majorly include unexpected machine failures that can lead to disruptions in operations, financial losses and safety hazards. Reactive and preventive maintenance which are a part of traditional maintenance strategies, often fall short in mitigating these risks due to inefficiency in predicting failures in real time. So, there is a need for an advanced approach that leverages machine learning and artificial intelligence to detect the anomalies and forecast potential failures before they occur. Industry 5.0 aims to handle repetitive tasks using AI and machine learning, while also working with humans to support them in thinking and decision making[1].
This research focuses on utilizing Deep Learning (DL) and Federated Learning (FL) algorithms to analyze sensor data and to detect faults which makes this paper focus on AI-driven predictive maintenance systems for industrial machines. According to the productivity increase, the production costs will be lowered down and therefore, competitiveness of the market will see an increase[3]. This study is significant in a way that it explores the integration of Federated Learning algorithms that never compromises with data privacy. In a nutshell, it ensures real-time fault detection along with maintaining data security across multiple industrial locations. Machine Learning uses its algorithms that are adaptive to enhance the system resilience with the prediction of potential failures. Supporting this, IoT enables communication across different industrial environments and boosts the effectiveness of proactive and condition-based maintenance with real time data collection and analysis[5]. Machine learning algorithms offer significant advantages by analyzing large volumes of data to detect patterns, which help reduce defects and maintain product consistency[7].
Our system is designed for machines that are deployed in multiple buildings , with each machine having four sensors on it - vibrational sensor, temperature sensor, humidity sensor and pressure sensor, which continuously monitor the operational conditions. The collected data is then sent to a centralized storage system which helps in identifying faulty machines. The research systematically applies ten different models of DL and FL on the sample data set and predicts machine failures. The workflow involves data preprocessing, feature extraction, model training, threshold-based fault prediction and warning generation so as to ensure timely maintenance action to prevent capital loss to the industries. By comparing the performances of different models including Multilayer Perceptron (MLP), Long Short Term Memory (LSTM), Convolutional Neural Networks (CNN), Transformers, Autoencoders, EfficientNetID, Adaptive FL, Hierarchical FL, Channel Separated CNN FL and Ensemble FL, we aim to determine the best and the most efficient approach for predictive maintenance. For the comparative effectiveness, the evaluation metrics and performance visualizations have been provided. 
This research majorly contributes to the advancements of intelligent predictive maintenance systems which enhance machine reliability, reducing downtime and hereby, optimizing operational efficiency. With the integration of federated learning in machine downtime prediction, we have addressed the major challenge related to data privacy and decentralized model training which paves way for more scalable and secure AI-driven maintenance solutions for addressing industrial problems. Federated Learning (FL) is a machine learning approach that trains data spread across numerous client devices in a distributed network, aiming to build a high quality model without the need to centralize the data[8].
1.1 Background
The tools and technologies that comprised Industry 4.0 evolved into Industry 5.0 in 2021. Now, it represents a major change in industrialisation. Whereas Industry 4.0 focused primarily on digitising manufacturing processes using technologies like the Internet of Things (IoT) and cyber physical systems, Industry 5.0 emphasises human-advanced technology, especially artificial intelligence (AI), working together. As a result, there are benefits to all aspects of efficiency through improved workflow and process management on-site. Industry 4.0 has transitioned to Industry 5.0 with the principle that increased human and machine collaboration produces improved synergy between these two elements in a single environment. Each party will benefit from the other; humans bring creativity, innovation, and critical thinking to the equation while machines bring speed and precision to the manufacturing process.

Industry 5.0 represents a data-driven digital transformation; it will be the next iteration of Industry 4.0 (after mass production) and provide additional ways to respond to the increasing need for mass personalization. Industry 5.0 includes three pillars of thought: resiliency, sustainability, and humanity. Together, these pillars embrace the well-being of people as well as economic development. Through this lens, digital transformations are driven by societal needs and will address societal issues that have arisen since the development of modern industries.This evolution understands that, although automation offers tremendous advantages, the future industries will benefit most when human skill is blended with technological prowess.

1.2 Research Questions
This research explores how the emerging technologies that come under Industry 5.0 - specifically IoT, Deep Learning and Federated Learning can increase the predictive maintenance in industrial machines. By addressing the key challenges, especially unexpected failures and data privacy, the study aims to implement fault detection on real-time sensor data along with a comparative study of various DL and FL models on the same data to show how FL is better in various ways. 
RQ1: What are the key challenges associated with traditional maintenance approaches in Industry 4.0 contexts, particularly regarding machine failures, operational inefficiencies, and data security?
RQ2: How can IoT, DL, and FL technologies be leveraged to develop intelligent, secure, and real-time predictive maintenance solutions that address these challenges?
RQ3: How can a fault detection and failure prediction system be designed and implemented using DL and FL models on sensor data (e.g., vibration, temperature, humidity, and pressure) from machines across multiple distributed locations?
RQ4: How do various DL and FL models (e.g., MLP, LSTM, CNN, Transformers, Autoencoders, EfficientNet1D, Adaptive FL, Hierarchical FL, Channel-Separated CNN FL, and Ensemble FL) compare in terms of accuracy, precision, recall, and F1-score for predictive maintenance, and what factors explain differences in their performance (e.g., FL's benefits in privacy and efficiency)?

2. Literature review

In recent years, the predictive maintenance of industrial machinery using the sensor data has gathered significant attention. With the increasing complexity in the industrial systems, traditional methods of maintaining industrial machines are falling short in handling sensor patterns and predicting faults in them with a high accuracy. The emerging technologies such as Deep Learning (DL) and Federated Learning (FL) which are a part of Industry 5.0 technologies have shown a great improvement in accurate machine fault prediction by gathering IOT based sensor data from machines. FL algorithms are best in providing data privacy while performing the same. This section reviews key studies and methodologies that are relevant to our research. Ejjami and Boussalham[7] put emphasis on the fact that predictive maintenance not only reduces unplanned downtime but also helps in enhancing equipment life and minimizes human error by automating monitoring through AI-driven approaches. A major shift in fault detection has emerged through deep learning. As demonstrated by Murtaza et al. [5], CNN and LSTM models effectively capture various dependencies like spatial and temporal dependencies in vibrational signals that are collected over long periods, while on the other hand, hybrid architectures integrating autoencoders and transformers offer even higher predictive performance. With the rapid expansion of IoT-generated data, privacy concerns have positioned Federated Learning as a vital Industry 5.0 technology. Li et al.[8] explains how FL enables training on decentralized data across multiple devices without sharing any raw sensor data. Also, further improvements such as  Hierarchical FL, Adaptive FL, and channel-separated CNN-FL enhance model convergence under heterogeneous distributions. Recent studies also show that the combination of DL and FL produces robust-hybrid frameworks that preserve privacy with ensemble FL improving the fault-classification accuracy. In addition to this, generating maintenance alerts when fault probability exceeds the threshold value, with the help of dynamic and probability based thresholding techniques, outperform static thresholds by offering early adaptive fault detection. Together, the contributions of Murtaza et al. [5], Ejjami and Boussalham [7], and Li et al. [8] form a comprehensive foundation for modern, secure, and highly accurate predictive maintenance systems aligned with Industry 5.0 advancements.
2.1 Research Gaps

While Industry 5.0 builds upon technologies overlooked in Industry 4.0, the existing research mainly emphasises on Deep Learning to predict machine downtime and often neglect certain crucial aspects such as data privacy, adaptability, and the decentralized nature of industrial environments.
Many frameworks suffer from insufficient empirical validation [4], limited real-time integration [6], and theoretical overreach without practical case studies [7] which is also discussed in the table 1.
To handle complex and heterogeneous industrial data is difficult because the current methods often assume uniform training datasets and so fail to scale dynamic real-world conditions. This leads to the development of underperforming models.
The capacity of Federated Learning (FL) in order to provide privacy-preserving predictive maintenance remains unexplored because the current approaches struggle with data imbalance and non-IID distributions, affecting the model reliability.
Additionally, static anomaly detection thresholds are unable to adapt to changing operational conditions that cause false alarms and reduced system trust where most studies still lack real-time, decentralized deployment capabilities.
To address such limitations, this research paper proposes a real-time Federated Learning framework, a dynamic thresholding for accurate fault prediction that works more accurately under environmental variability. In addition to this, it ensures data privacy without compromising predictive accuracy. There are some previous research papers based on machine fault prediction using different algorithms and techniques. Table 1 depicts a detailed comparison of all the research papers covering the methodology followed along with the merits and demerits in them. These have helped us approach the problem in a very detailed manner. 

			Table 1 Comparative Analysis of Related Work

Source
Purpose
Proposed solution / method
Merits
Demerits
Mildrend et. al. [3]
Using method engineering to present a method to eliminate downtime along with improvement of manufacturing cells.
Elaborating man-machine diagrams to recognize dead times by analyzing lathe and grinding processes.
Reduction of downtime by 41% and only 50% labour is required.



To analyze and improve human-machine is the main focus while other enhancing factors for machine operation time are not considered.
Chowdury M. L. Rahman [2]
To evaluate the production performance of semi-automated manufacturing companies located in Bangladesh and the effect of Total Productive Maintenance
To point-out the most affecting factors leading to downtime hierarchically using Pareto and statistical analysis of downtimes.



Contributes valuable insights into TPM effectiveness, downtime reduction strategies, and data-driven decision-making in manufacturing maintenance
The study is based on only one semi-automated manufacturing company in Bangladesh, which limits the applicability of the findings to other industries, automation levels, and regions.
Akundi et. al. [1]
To identify and look for research trends related to Industry 5.0 with the help of tools and techniques. 
To use text mining tools and techniques to find resemblance and determine future research directions in order to lead the shift towards Industry 5.0.



Advanced and future research on the impact of the manufacturing landscape are identified.
The study is from 2016-2022 with only 196 abstracts which may not be sufficient to draw a conclusion keeping in mind how Industry 5.0 has evolved after these years.
Pizon and Gola [4]
To identify the status and direction of evolving human-machine relationship in the context of Industry 5.0
To point out the direction of evolution of human-machine relationship and its status with respect to Industry 5.0.
Understanding human-machine interaction development and provides a strategic roadmap for Industry 5.0
Lack of empirical validation, subjective framework, and absence of implementation details
Murtaza et. al. [5]
to examine the integration of Industry 5.0 principles with advanced predictive maintenance and condition monitoring 
To understand the combination of Industry 5.0 principles and advanced predictive maintenance with condition monitoring.
The six-layered framework and case study validation enhance its real-world applicability, making it a valuable resource for industry.
There is no emphasis on cross-industry collaboration along with lack of in-depth examination into the application of edge computing for PdM and CM
Kiangala and Wang [6]
Enhancing troubleshooting and predictive maintenance in industrial settings using AI chatbot-based Human-Machine Interface (HMI)
Experimenting hybrid AI model and generative AI chatbot HMI on machines.
a promising AI-driven HMI for Industry 5.0 with strong potential in troubleshooting, predictive maintenance, and SME adoption
Needs broader testing, real-time data integration, improved scalability, and discussion on cost-effectiveness.
Ejjami and Boussalham [7]
Effect and applications of Artificial Intelligence in manufacturing and machine maintenance.
Emphasising on the supply chain optimization, quality assurance and predictive maintenance applications of AI with the help of academic publications, industry reports and scholarly articles.
Acknowledges the critical issues of bias, transparency, and fairness in AI making decisions and suggests mitigation strategies.


The study heavily relies on theoretical discussions and does not provide concrete experimental data or real-world case studies to support its claims
Proposed Work
To prove that FL models are more efficient in predicting industrial machine downtime.
Comparative analysis of DL and FL models to predict the downtime of industrial machines using centralized data where FL models work more efficiently. 
FL models give more accuracy for centralized data along with maintaining data privacy.
-






3. Methodology
To conduct this study, we reviewed Industry 4.0, predictive maintenance, traditional methods, IoT and AI models like Deep Learning and Federated Learning. This review helped us in identifying the existing gaps in traditional approaches and understanding how new and smart 
Fig. 1 Research Methodology


technologies can be used to enable early fault detection in industrial machinery. This paper investigates machines like pumps, turbines and compressors, each 
with four IoT based sensors - temperature, humidity, pressure and vibration- to gather real time data. This collected data is processed and analysed using ten different AI models out of which there are five Deep Learning and five Federated Learning algorithms. A comparative analysis is then carried out using the Classification-Evaluation metrics to determine the most effective algorithm for predictive maintenance. This methodology shows us that FL algorithms not only provide data privacy and security, but also offer high fault prediction performance. In spite of such promising results, several challenges were encountered like Computational Complexity: because some DL models require high processing power that makes deployment difficult. Additionally, some Federated Learning Constraints were found where decentralised training across various industry locations causes model divergence for FL models. This research has far reaching implications for both: improvements to smart manufacturing systems through adaptive intelligence and as an enabler of the Industry 5.0 revolution. It illustrates how AI-based techniques increase the reliability of manufacturing assets. Traditional methods typically do not provide real-time predictions of asset failures, resulting in unplanned downtime and increased operational costs. This research is therefore supporting the development of scalable and secure maintenance systems using Industry 5.0 technologies. As shown in fig 1, the Internet of Things (IOT) uses sensors to record key factors such as temperature, pressure, vibration, and humidity to help track various machines in real-time. Data captured from IOT sensors are collected continuously and provide up-to-date information on how well a machine functions. With AI-powered predictive analytics, machine failure can be anticipated before it occurs. The results are derived using two methods: Deep Learning (DL) and Federated Learning (FL). Deep learning mainly relies on centralised data processing, which presents privacy and scalability issues but it provides high prediction accuracy by spotting intricate patterns in massive datasets. While on the other hand, Federated Learning makes it possible to train models that are distributed across several devices without exchanging raw data that provides data privacy along with predictive accuracy being maintained. The accuracy, scalability, and privacy preservation capabilities of both methods are then compared. The outcomes clearly show the benefits of embedding FL's privacy and decentralisation features with DL's analytical strength. 
Ultimately, the integrated framework provides a comprehensive maintenance solution for Industry 5.0, offering real-time monitoring, accurate fault prediction, and secure, scalable data management for next-generation industrial environments.
3.1 Proposed Work
This research primarily focuses on the predictive maintenance of the industrial machines that are deployed across multiple buildings. Each building consists of several rooms and each room has multiple machines. We have every machine that is equipped with four distinct types of sensors which is shown below in Fig. 2:
Vibrational Sensor
Humidity Sensor
Temperature Sensor
Pressure Sensor


				Fig. 2 Types of Machines and Sensors

These sensors generate real-time data streams after continuously monitoring machine parameters. This collected data is then systematically transmitted to a centralized storage system which is separate for each building. In addition to the sensor data, this database also records the machine ID and room ID which ensures efficient tracking and easy identification of the faulty machines. The centralized storage enables effective management and retrieval of the sensor information for further processing to detect the faultiness.

3.2 Algorithms 
Proposed work implemented various algorithms on the dataset. Detailed description is given in the table below:
Table 1. Algorithms used in proposed work

        Algorithm
     definition
             reference 
Multilayer Perceptron (MLP)
 An MLP (Multilayer Perceptron) is one of the artificial neural networks which use multiple layers of neurons to learn patterns in data. It is structured in layers, where each layer contains multiple nodes, and every node in this layer is connected to all the other nodes in the next layer.


An MLP consists of at least three layers, an input layer, many hidden layers and an output layer [8].
Long Short Term Memory (LSTM)
It is one of the recurrent neural networks (RNN) in machine learning which is used to process, learn, and classify sequential data. However, in this paper, its effectiveness is also demonstrated in fault categorization.


LSTM can maintain time coherence and so, it is commonly used in speech recognition [9].
Autoencoder 
Autoencoders are another category of models effective in revealing complex data. They are like smart compression and decompression algorithms. For our project, we have implemented an elementary autoencoder to familiarize ourselves with how "normal" equipment patterns are like using readings from sensors such as pressure and temperature. It does this by compressing the input data down to a smaller, more compact form (the encoding) and then attempting to reproduce the original data again from this compressed form (the decoding).


-
Convolution Neural Network (CNN)
CNNs are a popular kind of neural network that was inspired by human visual processing and created to handle grid-like data. They function as a specific subset of artificial neural networks.


Every single neuron in feature map of CNN layer is connected only with a small subset of neurons from the previous one[10].
Transformer 
This is an architecture of neural networks which are designed to convert the sequence of input into output series that promotes efficient handling of complex tasks.


Transformers do this by learning context and tracking relationships between sequence components[11].
EfficientNet1D
EfficientNet is a convolutional neural network architecture. Its goal is to increase the efficiency and accuracy of the model with the help of Compound Scaling Method. 


The EfficientNet Series surpasses previous network architectures in performance and efficiency by methodically adjusting the network’s depth, width, and resolution[12].
Channel Separated CNN FL
In channel separated CNN FL, the convolution operations are applied separately to different channels rather than to the combined ones at the beginning.


It reduces the computational cost while maintaining performance and also improves privacy by avoiding the early channel fusion [13].
Hierarchical FL
Hierarchical FL represents and analyses complex relationships and patterns in data using a structured approach.


The main goal of using hierarchical Federated Learning (FL) is to decrease the communication costs involved in detecting the objects14].
Adaptive FL
Adaptive federated learning (FL) is a method that helps the devices to get used to global models on the basis of their local data.
AFL can dynamically adjust the large amount of data for the next training round based on the previous round’s energy consumption [15].
Ensemble FL
Ensemble federated learning is one of the ML techniques which uses the combination of multiple models to make predictions and preserve user privacy.


It is used to improve the classification performance with the combination of output from first level having original features to improve the results in next rounds [16].



3.3 Data Collection and Storage

The design of the proposed system is made to ensure efficient data flow and management. Fig. 3 describes the layout of the proposed work where the raw sensor data is gathered from the machines periodically and stored in a robust and scalable database. The storage infrastructure is designed in a way that it handles multi-sensor data efficiently, thereby ensuring structured organization with relevant identifiers for seamless access and processing.
The collected data includes :
Timestamped readings from all the four sensors
Machine ID to uniquely identify the machine
Room ID to track the machine’s room

				

				Fig. 3 Layout of the Model

This structured dataset is the basic foundation for machine fault detection and predictive maintenance and so the system ensures data integrity, security and accessibility , allowing real-time analytics and decision-making. The system also provides data validation by eliminating noise, handling missing values and normalizing readings before feeding it to the predictive models. This design also ensures high reliability and low latency in fault diagnosis, hence, making it suitable for real world industry deployment.
3.4 Fault Detection using Deep Learning and Federated Learning Algorithms
To analyse and predict the machine failures, we process the data through a total of ten models and then check the accuracy. These ten models are a combination of Deep Learning (DL) and Federated Learning (FL) algorithms which process the sensored data being stored so as to detect irregularities and predict the expected faults before they actually occur.. Fig. 4 shows all the steps that are being executed for the DL and FL algorithms used to extract the results for machine downtime fault prediction :


Data processing : The sensor data is initially cleaned, normalized and then transformed for consistency and quality assurance.
Feature Extraction : The key indicators and the critical parameters of the machine such as vibration frequency, humidity levels, temperature variations and pressure fluctuations are extracted.
Model Training : To recognize failure patterns and deviations from the normal operational behavior, a set of DL and FL models are trained on historical data.
Threshold-Based Fault Prediction : A prediction threshold value of 0.8 is applied to classify machines as normal or potentially faulty and if a machine’s probability score exceeds this threshold value, then it is flagged for inspection.
Warning Generation : The system generates the alerts that contain machine ID and room ID before the failure occurs and so it leads to early identification and proactive maintenance.


				Fig. 4  Algorithmic Workflow

The system is designed to continuously improve its capability to predict faults through iterative cycles. The models adapt to the evolving machine conditions after some new sensor data is collected which ensures that the framework remains accurate and becomes scalable over time. The adaptive nature of the model helps in enabling predictive maintenance that reduces unexpected downtimes and improves overall efficiency.

3.5 Implementation
In order to draw the results, we have proceeded with the above workflow on a dataset.
The whole process is divided into four main stages :
Data Preprocessing : The process begins with loading the sensor data of the machines from the CSV files and then undergoes multiple preprocessing steps in order to ensure consistency and quality. The sub-methods performed are shown below in Fig. 5 :
Feature Preprocessing : 
One-Hot Encoding : The categorical variables that represent machine types are converted into numerical form.
Standardization and Scaling : To ensure uniformity across the dataset, the numerical sensor readings (vibrational , humidity, temperature and pressure) are standardized.
Feature Concatenation :  All the processed features are merged to create an extensive dataset.
Data Splitting by Location : The dataset is split by physical location since the machine exists in different buildings and rooms.
Client Dataset Creation :  To prepare the data for Federated Learning implementations, the pre-processed data is now organized into client datasets.

				Fig. 5 Data Processing in the Proposed Model

After this phase, the refined datasets are now ready for the model training and evaluation. This step ensures that the data is clean and well-structured to train the models. Since we have both centralized and federated learning frameworks, we organize the data into client-specific datasets that also helps the system to enable decentralized model training while preserving data privacy. 
Mathematical Representation of Models: To have the comparison between the different models and check the model with the maximum accuracy, the models are categorized into Deep Learning and Federated Learning models.

DEEP LEARNING MODELS:

Multilayer Perceptron (MLP):
y=σWout⋅ReLUWhidden⋅x+bhidden+bout┤) 
Where:
x refers to an input vector.
Whidden,bhidden  are weights and biases of hidden layers.
ReLUz=0,zReLU activation function.
Wout,bout  are weights and biases of the output layers.
z=11+e-z refers to a Sigmoid activation function.
y  is the output (predicted value).




Long Short-Term Memory (LSTM): 
ht,Ct=LSTMxt,ht-1,Ct-1
y=σWohT+bo
Where:
x  refers to input time step t.
ht,Ct refer to the hidden state and to the cell state at time t.
LSTM  represents LSTM cell function.
ht-1,Ct-1 refer to the hidden and cell states from the previous time step.
hT is the final hidden state after processing the sequence.
Wo,bo are weights and biases of a final layer.
 is the Sigmoid activation.
y  is the output.


Simple Autoencoder: 
z=ReLUWenc2⋅ReLUWenc1⋅x+benc1+benc2┤) 

x=Wdec2ReLUWdec1z+bdec1+bdec2┤) 

x=DecoderEncoderx
Where:
x  refers to the input.
z  is encoded representation.
x is the reconstructed output.
Encoder and Decoder represent their respective transformations.
Wenci,benci , Wdeci,bdeci are weights and biases.
ReLU  and  are the ReLU and Sigmoid activations.


Basic CNN : 
y=σWfc⋅FlattenMaxPoolReLUConv1Dx,Wconv+bconv+bfc┤) 
Where:
x  is 1D input sequence (or tensor).
Conv1Dx,Wconv+bconv  refers to the convolution function.
ReLU  activation function.
MaxPool  is max pooling operation.
Flatten  converts the pooled output into a vector.
Wfc,bfc  are weights and biases of a layer.
 is Sigmoid activation.
y  is the output.


Simple Transformer : 
H=TransformerEncoderLinearx

y=σWout⋅MeanH+bout 

Where:
x refers to input sequence.
Linearx is initial linear projection.
TransformerEncoder  applies self-attention and feed-forward layers.
H  is the output of the encoder.
MeanH computes the mean of all encoder outputs.
Wout,bout  are weights and biases of the output layer.
 is Sigmoid activation.
y  is the output.


FEDERATED LEARNING MODEL :
Statement
Notation
Global parameters at round (r)
Wglobalr
Client’s parameters at round (r)
Wir
Local dataset of client (i)
Di
Loss function for client (i)
LW;Di
Number of local training epochs per round
E
Number of participating clients
N


Step 1: Initialization
Wglobal0Initialize random weights 
Step 2: Client Training
Each client receives the global model:
Wclient,i0=Wglobalr-1 
Each client updates its parameters using gradient descent:
Wclient,ie=Wclient,ie-1-η∇Wclient,ie-1;Di) 
Where  refers to the learning rate and ∇L  refers to the gradient of loss function.


After E  local epochs:
Wclient,ir=Wclient,iE 
Step 3: Server Aggregation (Federated Averaging)
Wglobalr=1Ni=1NWclient,ir 


Fig. 6 Federated Learning process workflow

At each round r, the server sends a global model to all the clients. The clients update locally after training on its dataset  for epochs and then update the model. Also, the server aggregates updates. After R rounds, WglobalR
   is the final trained global model.
Training Process :  After defining the models, they undergo a training process as illustrated in Fig.6. To ensure that the evaluation is unbiased, the dataset is first divided into training and testing subsets that allow models to learn from historical data. The training process begins with the definition of specific training functions. The generic training functions are employed for standard deep learning architectures and the specialized training functions are designed for autoencoders to identify anomalies in the sensor data. For the continuous assessment of performance and optimized parameters during training, model evaluation functions are also integrated. Fig. 7 illustrates the whole process, the model undergoes.



					Fig. 7 Training Process 

Following this process, both the models are trained on the processed datasets. Deep Learning models are trained in a centralized environment in order to capture complex patterns and relationships
Evaluation : After the training process, the models are evaluated on multiple performance metrics.
Results Visualization : Data driven insights are generated from the model outputs.
Accuracy Calculation :  The models’ predictive accuracy is computed.
Model Comparison : 
Boxplot Comparison is done to analyse variability in the performance.
DL vs. FL Average Performance comparison is performed to assess the effectiveness.
Overall Performance Evaluation determines the most optimal approach for predictive maintenance.
4. Results and discussions
Our experimental evaluation shows the effectiveness of various models including deep learning and federated learning, in achieving high classification accuracy and precision. We have implemented a Classification-Evaluation matrix for all the ten models and have drawn all the results shown in the graphs shown below. The graphs below show the comparison among all the ten models based on the four factors of the Classification-Evaluation Matrix. Table 2 shows the percentage comparison among the models that clearly shows which model is better in what aspect. With the overall analysis, it is hereby proved that, for centralized data gathered from the industrial machines, FL-based models are better in predicting the downtime and also provide data privacy. Among all the FL models, Channel Separated FL are highly effective for this task proving their potential for secure and decentralized model training.

				Table 2 Classification-Evaluation Matrix

Model
Accuracy


Precision


F1 score


Recall


Hierarchical Federated Learning (FL)
98.44 %
99.24 %
91.55 %
84.97 %
Efficient Net 1D (FL)
98. 37 %
95.71 %
91.47 %
87.58 %
Adaptive Federated Leaning (FL)
98.24 %
98.46 %
90.46 %
83.66 %
Channel Separated Federated Learning (FL)
98.11 %
98.44 %
89.68 %
82.35 %
Simple Transformer (DL)
98.05 %
98.43 %
89.29 %
81.70 %
Ensemble Federated Learning (FL)
97.46 %
100 %
85.39 %
74.51 %
Basic Convolutional Neural Network CNN (DL)
96.55 %
94.64 %
80 %
69.28 %
Multi-Layer Perceptron MLP (DL)
95.90 %
100 %
74.07 %
58.82 %
Long Short Term Memory LSTM (DL)
93.42 %
100 %
50.73 %
33.99 %
Simple Autoencoder (DL)
90.75 %
100 %
13.41 %
7.19 %



ACCURACY :- Accuracy is the measure of how many predictions were correct out of all the predictions. Table 2 shows the accuracy comparison among all the ten models with Hierarchical Federated Learning being the model with best accuracy. Fig 8 shows the accuracy comparison among all the models and below is the formula describing how accuracy is calculated in a model :

Accuracy = tp+tn tp+tn+fp+fn


tp = True Positive (correctly predicted positive cases)


tn = True Negative (correctly predicted negative cases)


fp = False Positive (incorrectly predicted positive cases)


fn = False Negative (incorrectly predicted negative cases)


					Fig. 8 Accuracy Comparison
	

PRECISION :- Precision is the ratio of correctly predicted positive observations to the total number of predicted positive observations. It is a measure of how accurate the positive predictions are. Fig. 9 shows the comparative graph of all the DL and FL models.
Below is the formula describing how precision is calculated in a model :
Precision=tptp+fp
Where,
tp = Total Positive
fp=  False Positive


					Fig. 9 Precision Comparison


RECALL :- Recall refers to the ratio of all correctly predicted positive observations to all actual positive observations.
The comparison of all the ten models’ recall is shown in Fig. 10 with Efficient Net 1D resulting in the best recall percentage.  Below is the formula describing how recall is calculated in a model :


Recall = tptp+fn
Where, 
TP  = Total Positive
FN  = False Negative



					Fig. 10 Recall Comparison

F1 SCORE :- The F1 score combines precision and recall into a single metric, offering a more balanced evaluation of a model’s performance, especially when the classes are imbalanced.  Fig. 11 has the comparison between DL and FL models and Hierarchical Model has the best F1 Score.  Below is the formula describing how F1 Score is calculated in a model :
F1 Score=2×Precision×RecallPrecision+Recall}} 




Fig. 11 F1 Score Comparison

This study primarily focuses on the integration of Deep Learning (DL) and Federated Learning (FL) algorithms for AI-driven predictive maintenance in industrial settings. By examining sensor data from numerous machines located in various locations, we have evaluated how well different models detect faults and forecast failures. This study demonstrates how AI can increase machine dependability, decrease downtime, and improve maintenance plans.
Ten AI models were compared, shedding light on the advantages and disadvantages of various strategies. Due to their capacity to identify spatial patterns in sensor data, LSTM and CNN stood out among the DL models in terms of fault detection accuracy. Transformers performed well in the areas where long-range dependencies need to be considered while autoencoders are useful in detecting anomalies. EfficientNet1D has strong feature extraction capabilities along with minimal computational overhead. As for FL models, Adaptive FL and Ensemble FL outperformed other decentralized models by maintaining high predictive accuracy and preserving data privacy along with it. Hierarchical FL works efficiently in balancing local and global model updates while Channel Separated CNN FL handles multi-sensor data quite productively. However, some FL models have reduced performance because of lack of centralized data aggregation.

5.1 Implications
Our approach enables early fault detection, thereby, allowing maintenance of the machinery before the industry encounters any loss and will eventually improve overall production efficiency. The combination of Deep Learning and Federated Learning enable the systems to effectively detect real-time fault and predict expected failures and at the same time preserve data privacy through the decentralized model training. This guarantees increased operational efficiency, decreased downtime, and improved machine reliability in dispersed industrial settings. Common industrial sensors and IoT infrastructure support the system's scalable nature, enabling smooth deployment across numerous locations. The industries are assisted in selecting the most effective predictive maintenance solution based on their requirements by the comparative study of ten DL and FL models. All things considered, this study advances the creation of safe, intelligent, and scalable AI-driven maintenance frameworks that support resilient, competitive, and cost-effective manufacturing operations and are in line with Industry 5.0 objectives. The primary ramifications of our suggested work are as follows: This study backs a successful transition from conventional machine failure detection techniques to predictive maintenance driven by artificial intelligence. This reduces machine downtime by enabling industries to identify machine faults in real time and before they actually happen. Training on decentralised data without moving it to the central server is made possible by federated learning algorithms. This aids in maintaining data privacy, which is crucial for sectors that handle sensitive operational data. Businesses gain from improved fault detection and early failure prediction outcomes. This prevents production halts and lowers maintenance and repair expenses. The combination of deep learning and federated learning inclines perfectly with the goal of Industry 5.0 that focuses on human expertise and AI’s power. AI handles repetitive and heavy tasks and humans focus on complex decisions. The use of common sensors and real-time data collection makes the solution practical and deployable.
6. Conclusion and future work
The addition of AI in prediction of machine maintenance has a lot of industrial applications, these span from optimizing maintenance schedules to extending the lifespan of machinery. AI is effective in improving workspace safety, since it can predict potential failures and allow for pre-emptive actions. There are several areas that could further enhance the effectiveness of AI in this field. For example, expanding sensor coverage to include electrical signals could provide better fault detection. Implementing edge computing could also reduce dependency on centralized systems, enabling faster data processing and reducing potential network latencies, since the model would be run locally. AI driven strategies could also be developed to offer better and more complete maintenance suggestions based on real time predictions, allowing for more dynamic and adaptive approaches to maintenance. By implementing these capabilities of AI systems, industrial operations can be transformed. This would ensure greater efficiency, enhance safety and result in significant long term cost savings.

References

Aditya Akundi, Daniel Euresti , Sergio Luna, Wilma Ankobiah , Amit Lopes and Immanuel Edinbarough; State of Industry 5.0—Analysis and Identification of Current Research Trends; published in MDPI on 17 February 2022

Chowdury M. L. Rahman ; Assessment of Total Productive Maintenance implementation in a semi- automated manufacturing company through downtime and mean downtime analysis; presented in Proceedings of the 2015 International Conference on Industrial Engineering and Operations Management Dubai, UAE, March 3 – 5, 2015

Montoya-Reyes, Mildrend; González-Angeles, Alvaro; Mendoza-Muñoz, Ismael; Gil- Samaniego-Ramos, Margarita; Ling-López, Juan; Method engineering to increase labor productivity and eliminate downtime; published in Journal of Industrial Engineering and Management (JIEM)

Jakub Pizon and Arkadiusz Gola; Human–Machine Relationship—Perspective and Future Roadmap for Industry 5.0 Solutions; published in MDPI on 1 February, 2023

Aitzaz Ahmed Murtaza, Amina Saher, Muhammad Hamza Zafar, Syed Kumayl Raza Moosavi , Muhammad Faisal Aftab , Filippo Sanfilippo; Paradigm shift for predictive maintenance and condition monitoring from Industry 4.0 to Industry 5.0: A systematic review, challenges and case study; published in Results in Engineering.

Kahiomba Sonia Kiangala, Zenghui Wang; An experimental hybrid customized AI and generative AI chatbot human machine interface to improve a factory troubleshooting downtime in the context of Industry 5.0; published in The International Journal of Advanced Manufacturing Technology (2024) on 3 April, 2024

Rachid Ejjami, Khaoula Boussalham; Industry 5.0 in Manufacturing: Enhancing Resilience and Responsibility through AI-Driven Predictive Maintenance, Quality Control, and Supply Chain Optimization; published in International Journal for Multidisciplinary Research (IJFMR)

Zhinong Li ORCID, Zedong Li, Yunlong Li ORCID, Junyong Tao, Qinghua Mao and Xuhui Zhang ORCID; An Intelligent Diagnosis Method for Machine Fault Based on Federated Learning, Published in MDPI on 21 December, 2021.

Pier Francesco Orrù, Andrea Zoccheddu, Lorenzo Sassu, Carmine Mattia, Riccardo Cozza and Simone Arena; Machine Learning Approach Using MLP and SVM Algorithms for the Fault Prediction of a Centrifugal Pump in the Oil and Gas Industry; Presented at MDPI , accepted on 9 June, 2020 

Russell Sabir, Daniele Rosato, Sven Hartmann and Clemens Gühmann; LSTM based Bearing Fault Diagnosis of Electrical Machines using Motor Current Signal; presented at 2019 18th IEEE International Conference on Machine Learning and Applications 

Long Wen , Xinyu Li , Liang Gao , Member, IEEE, and Yuyan Zhang ; A New Convolutional Neural Network-Based Data-Driven Fault Diagnosis Method;  IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS, VOL. 65, NO. 7, JULY 2018 

Guoqiang Li , Meirong Wei, Haidong Shao , Senior Member, IEEE, Pengfei Liang , and Chaoqun Duan; Wavelet Knowledge-Driven Transformer for Intelligent Machinery Fault Detection With Zero-Fault Samples; IEEE SENSORS JOURNAL, VOL. 24, NO. 21, 1 NOVEMBER 2024 

DI WU 1 , YONG HONG1 , JIE WANG2 , SHAOJUN WU2 , ZHIHAO ZHANG1 , AND YIZHANG LIU1; EfficientNet-b0-Based 3D Quantification Algorithm for Rectangular Defects in Pipelines; presented at IEEE; published in IEEE MAGNETICS SOCIETY SECTION published on 3 January, 2025.

Jun Lin; Jin Ma; Jianguo Zhu; Hierarchical Federated Learning for Power Transformer Fault Diagnosis; published in IEEE Transactions on Instrumentation and Measurement. Published on 22 August 2022

Ibrahim Ali Alnajjar, Laiali Almazaydeh, Ali Abu Odeh, Anas A. Salameh, Khalid Alqarni, Anas Ahmad Ban Atta; Anomaly Detection Based on Hierarchical Federated Learning with Edge- Enabled Object Detection for Surveillance Systems in Industry 4.0 Scenario; International Journal of Intelligent Engineering and Systems, Vol.17, No.4, 2024

Fangming Deng , Member, IEEE, Ziqi Zeng , Wei Mao , Baoquan Wei , and Zewen Li; A Novel Transmission Line Defect Detection Method Based on Adaptive Federated Learning; published in IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT, VOL. 72, 2023 

S. Patil* a , V. Phalleb; Fault Detection of Anti-friction Bearing using Ensemble Machine Learning Methods; published in IJE TRANSACTIONS B: Applications Vol. 31, No. 11, (November 2018) 1972-1981 

Emanuele Principi, Damiano Rossetti, Stefano Squartini, Senior Member, IEEE, and Francesco Piazza, Senior Member, IEEE; Unsupervised Electric Motor Fault Detection by Using Deep Autoencoders; published in IEEE/CAA JOURNAL OF AUTOMATICA SINICA, VOL. 6, NO. 2, MARCH 2019

Pangun Park , Piergiuseppe Di Marco , Hyejeon Shin and Junseong Bang; Fault Detection and Diagnosis Using Combined Autoencoder and Long Short-Term Memory Network; published in MDPI on 23 October, 2019

Jinyang Jiao , Ming Zhao  , Jing Lin , Kaixuan Liang; A comprehensive review on convolutional neural network in machine fault diagnosis

SYAHRIL RAMADHAN SAUFI, ZAIR ASRAR BIN AHMAD,  MOHD SALMAN LEONG AND MENG HEE LIM; Challenges and Opportunities of Deep Learning Models for Machinery Fault Detection and Diagnosis: A Review; Published in IEEE Access on August 29, 2019

SHEN ZHANG (Student Member, IEEE), SHIBO ZHANG (Student Member, IEEE), BINGNAN WANG (Senior Member, IEEE), AND THOMAS G. HABETLER (Fellow, IEEE); Deep Learning Algorithms for Bearing Fault Diagnostics—A Comprehensive Review; published in IEEE access on February 10, 2020

Bingbing Hu, Jiahui Tang, Jimei Wu and Jiajuan Qing; An Attention EfficientNet-Based Strategy for Bearing Fault Diagnosis under Strong Noise; Published in MDPI on 31 August 2022

MOHAMMED NASSER AL-ANDOLI, SHING CHIANG TAN , KOK SWEE SIM , (Senior Member, IEEE), MANJEEVAN SEERA , AND CHEE PENG LIM ; A Parallel Ensemble Learning Model for Fault Detection and Diagnosis of Industrial MachineryA Parallel Ensemble Learning Model for Fault Detection and Diagnosis of Industrial Machinery; published in IEEE Access on 14 April, 2023

Xiaoding Wang , Sahil Garg , Member, IEEE, Hui Lin , Jia Hu , Georges Kaddoum , Senior Member, IEEE, Md. Jalil Piran , Senior Member, IEEE, and M. Shamim Hossain , Senior Member, IEEE; Toward Accurate Anomaly Detection in Industrial  Internet of Things Using Hierarchical  Federated Learning ; published in IEEE INTERNET OF THINGS JOURNAL, VOL. 9, NO. 10, MAY 15, 2022

Ahlam Mallak and Madjid Fathi; Sensor and Component Fault Detection and Diagnosis for Hydraulic Machinery Integrating LSTM Autoencoder Detector and Diagnostic Classifiers; Published in MDPI on 9 January, 2021

Russell Sabir, Daniele Rosato, Sven Hartmann and Clemens Gühmann; LSTM based Bearing Fault Diagnosis of Electrical  Machines using Motor Current Signal; published in 2019 18th IEEE International Conference on Machine Learning and Applications (ICMLA)

Walter H. Delashmit and Michael T. Manry; Recent Developments in Multilayer Perceptron Neural Networks; presented at Proceedings of the 7th Annual Memphis Area Engineering and Science Conference MAESC 2005

YAN WANG , HUA DING, AND XIAOCHUN SUN; Residual Life Prediction of Bearings Based on SENet-TCN and Transfer Learning; published in IEEE Access on 18 November 2022.

Mohammed Alenezi, Fatih Anayi, Michael Packianather and Mokhtar Shouran; Enhancing Transformer Protection: A Machine Learning Framework for Early Fault Detection; Published in MDPI on 8 December, 2024

Guoqiang Li , Meirong Wei, Haidong Shao , Senior Member, IEEE,  Pengfei Liang , and Chaoqun Duan; Wavelet Knowledge-Driven Transformer for Intelligent Machinery Fault Detection  With Zero-Fault Samples; published in IEEE SENSORS JOURNAL, VOL. 24, NO. 21, 1 NOVEMBER 2024
Xueyi Zhang , Liang Ma , Kaixiang Peng , Chuanfang Zhang , Muhammad Asfandyar Shahid ; A cloud–edge collaboration based quality-related hierarchical fault detection framework for large-scale manufacturing processes; published in Expert Systems with Applications Volume 256, 5 December 2024, 124909
Jiayang Liu ,Xiaosun Wang ,Shijing Wu ,Liang Wan ,Fuqi Xie; Wind turbine fault detection based on deep residual networks; published in Expert Systems with Applications Volume 213, Part B, 1 March 2023, 119102
Wei Zhang, Xiang Li, Qian Ding; Deep residual learning-based fault diagnosis method for rotating machinery; published in ISA Transactions Volume 95, December 2019, Pages 295-305
SHENGNAN TANG , SHOUQI YUAN , AND YONG ZHU; Deep Learning-Based Intelligent Fault Diagnosis Methods Toward Rotating Machinery; published in IEEE on December 30, 2019
YoudaoWanga , Yifan Zhaoa , Sri Addepalli; Remaining Useful Life Prediction using Deep Learning Approaches: A Review; presented in 8th International Conference on Through-Life Engineering Service
Giuseppe Ciaburro; Machine fault detection methods based on machine learning algorithms: A review; published in Mathematical and Bioscience Engineering on 10 August, 2022
Bo Luo , Haoting Wang , Hongqi Liu, Bin Li, and Fangyu Peng; Early Fault Detection of Machine Tools Based on Deep Learning and Dynamic Identification; published in IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS, VOL. 66, NO. 1, JANUARY 2019
Miao He and David He, Member, IEEE; Deep Learning Based Approach for Bearing Fault Diagnosis; published in IEEE TRANSACTIONS ON INDUSTRY APPLICATIONS, VOL. 53, NO. 3, MAY/JUNE 2017
Jiahao Du ,Na Qin , Deqing Huang, Yiming Zhang , and Xinming Jia; An Efficient Federated Learning Framework for Machinery Fault Diagnosis With Improved Model Aggregation and Local Model Training; published in IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 35, NO. 7, JULY 2024
Tarek Berghout , Mohamed Benbouzid , Toufik Bentrcia, Wei Hong Lim  and Yassine Amirat; Federated Learning for Condition Monitoring of Industrial Processes: A Review on Fault Diagnosis Methods, Challenges, and Prospects; published in MDPI on 29 December 2022
Weihua Li, Wansheng Yang , Gang Jin , Junbin Chen , Jipu Li , Ruyi Huang and Zhuyun Chen; Clustering Federated Learning for Bearing Fault Diagnosis in Aerospace Applications with a Self-Attention Mechanism; published in MDPI on 15 September 2022
Zehui Zhang , Cong Guan , Hui Chen , Xiangguo Yang , Wenfeng Gong and Ansheng Yang; Adaptive Privacy-Preserving Federated Learning for Fault Diagnosis in Internet of Ships; published in IEEE INTERNET OF THINGS JOURNAL, VOL. 9, NO. 9, MAY 1, 2022
Tao Wen , Xiaohan Chen , Dingcheng Zhang , Clive Roberts , and Baigen Cai; A Sequential and Asynchronous Federated Learning Framework for Railway Point Machine Fault Diagnosis With Imperfect Data Transmission; published in IEEE TRANSACTIONS ON INDUSTRIAL INFORMATICS, VOL. 20, NO. 6, JUNE 2024
