import React from "react";
import UserManagement from "../components/UserManagement";

const Users = () => {
  return (
    <div className="min-h-screen bg-gray-100">
      <div className="py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8">
          <h1 className="text-2xl font-semibold text-gray-900">
            User Management
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            Manage users, view analytics, and control user access
          </p>
        </div>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8">
          <div className="py-4">
            <UserManagement />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Users;
