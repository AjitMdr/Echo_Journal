import axios from "axios";

const axiosInstance = axios.create({
  baseURL: "http://100.64.193.236:8000/api",
});

// Add a response interceptor
axiosInstance.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Clear local storage
      localStorage.removeItem("access_token");
      localStorage.removeItem("refresh_token");
      localStorage.removeItem("user");

      // Redirect to login page
      window.location.href = "/login";
    }
    return Promise.reject(error);
  }
);

// Add a request interceptor to add the token to all requests
axiosInstance.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("access_token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

export default axiosInstance;
