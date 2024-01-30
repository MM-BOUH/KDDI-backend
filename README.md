# PHC prediction project backend

This is a prediction project backend that was done to test if the PHC data is enough to predict some healthcare features such as BMI, Temperature, Pulse rate. etc.

## Prerequisites

- [Python](https://www.python.org/) (3.8 or higher)
- [uvicorn](https://www.uvicorn.org/) (ASGI server)
- [fastapi](https://fastapi.tiangolo.com/) (as a python framework for the backend)

## Getting Started

1. Clone the repository:

   ```
   git clone https://github.com/MM-BOUH/prediction-project.git
   cd your-fastapi-project
   ```

2. Because of the size of the ML models, sometimes we use ngrok to expose the backend to the internet.
   1. Install the ngrok on your local machine,
   2. Get the authtoken from the platform and add it by this command:

```
   ngrok config add-authtoken <your_token>

```

3. Then, run it with this command:

```
    ngrok

```

4.  The backend is connected to the frontend that's deployed on netlify: [frontend on netlify](https://phc-prediction.netlify.app/).
5.  It's important to know that you need to change the backend url in the frontend github repo:[frontend on github](https://github.com/MM-BOUH/prediction-project).

6.  If you have enough storage on heroku for the size of the ML models, the backend is already deployed there, and can be found here: [backend on heroku](https://phc-demo-b3e9e6aa0c2b.herokuapp.com/)
