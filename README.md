# Titanic_prediction
[problem definition](problem) 

# solution
The solution has been developed using FastAPI for defining API endpoints.

using command 'docker-compose up' we will get API endpoint.

In order to access API endpoint, navigate to http://0.0.0.0:8000/docs 
![image](https://user-images.githubusercontent.com/72340440/168650553-e78cabfb-9a64-4bb3-b2fb-a1e8954bc5c8.png)


after lunch API we will see some parameters:
 1. Pclass: input [1,2,3] we have only 3 passenger class
 2. male: input (0 or 1)  this parameter is generated from parameter Sex where 1 means that passenger is male, 0 means female
 3. Age: input [0.42, 80] range of years from 0.42 to 80
 4. Siblings_Spouses: input [0, 8] range number of Siblings/Spouses from 0 to 8
 5. Parents_Children: input [0, 6] range number of Parents/Children from 0 to 6
 6. Fare: input [0, 512.3292] range of fare from 0 to 512.3292

After correctly filling all field API will give us probability if passenger Survived or not.
![image](https://user-images.githubusercontent.com/72340440/168650707-81186358-4159-4efe-a32e-5ed17ee552f2.png)
