import React, { createContext, useState, useContext, useEffect } from "react";
import { login as apiLogin, logout as apiLogout } from "../api/auth";
import { useNavigate } from "react-router-dom";

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [token, setToken] = useState(localStorage.getItem("access_token"));
  const navigate = useNavigate();

  useEffect(() => {
    // Check if user is already logged in
    const storedUser = localStorage.getItem("user");
    const accessToken = localStorage.getItem("access_token");

    if (storedUser && accessToken) {
      setUser(JSON.parse(storedUser));
      setToken(accessToken);
      setIsAuthenticated(true);
    }
    setLoading(false);
  }, []);

  const handleLogout = () => {
    // Clear all auth-related items from localStorage
    localStorage.removeItem("user");
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    setUser(null);
    setToken(null);
    setIsAuthenticated(false);
    navigate("/login");
  };

  const login = async (username, password) => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiLogin(username, password);
      setUser(response.user);
      setToken(response.access);
      setIsAuthenticated(true);
      return response;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    try {
      setLoading(true);
      await apiLogout();
      handleLogout();
    } catch (err) {
      setError(err.message);
      // Even if logout API fails, clear local storage and state
      handleLogout();
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const value = {
    isAuthenticated,
    user,
    loading,
    error,
    login,
    logout,
    token,
    handleLogout, // Expose handleLogout for token expiration
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
