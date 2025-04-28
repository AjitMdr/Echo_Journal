export const API_BASE_URL = "http://192.168.1.73:8000/api";

export const ENDPOINTS = {
  ADMIN: {
    DASHBOARD: {
      METRICS: `${API_BASE_URL}/admin/dashboard/metrics/`,
      TRENDS: `${API_BASE_URL}/admin/dashboard/trends/`,
      SUBSCRIPTION_ANALYTICS: `${API_BASE_URL}/admin/dashboard/subscription_analytics/`,
      USER_ANALYTICS: `${API_BASE_URL}/admin/dashboard/user_analytics/`,
      SUBSCRIPTION_STATS: `${API_BASE_URL}/admin/dashboard/subscription_stats/`,
    },
  },
};
