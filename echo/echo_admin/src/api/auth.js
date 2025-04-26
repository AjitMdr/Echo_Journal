import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://100.64.193.236:8000/api";

export const login = async (username, password) => {
  try {
    const response = await axios.post(`${API_URL}/auth/login/`, {
      username,
      password,
    });

    // Check if response is successful and user is SUPERADMIN
    if (response.status === 200 && response.data.user.role === "SUPERADMIN") {
      // Store tokens and user data in localStorage
      localStorage.setItem("access_token", response.data.access);
      localStorage.setItem("refresh_token", response.data.refresh);
      localStorage.setItem("user", JSON.stringify(response.data.user));

      return response.data;
    } else {
      throw new Error("Access denied. Only SUPERADMIN users are allowed.");
    }
  } catch (error) {
    if (error.response?.status === 401) {
      throw new Error("Invalid username or password");
    } else if (error.response?.status === 403) {
      throw new Error("Access denied. Only SUPERADMIN users are allowed.");
    } else {
      throw new Error(error.response?.data?.message || "Login failed");
    }
  }
};

export const logout = async () => {
  try {
    const token = localStorage.getItem("access_token");
    if (token) {
      await axios.post(
        `${API_URL}/auth/logout/`,
        {},
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
    }
    // Clear localStorage on logout
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    localStorage.removeItem("user");
  } catch (error) {
    throw new Error(error.response?.data?.message || "Logout failed");
  }
};
