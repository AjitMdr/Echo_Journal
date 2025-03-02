import 'package:flutter/material.dart';
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../../../services/profile/profile_service.dart';
import 'package:fluttertoast/fluttertoast.dart';

class ProfilePage extends StatefulWidget {
  @override
  _ProfilePageState createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  String? username;
  String? email;
  String? profilePictureUrl;
  bool isLoading = false;

  // To pick an image for profile picture
  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _loadProfileData();
  }

  // Load profile data from cache or API
  Future<void> _loadProfileData() async {
    setState(() {
      isLoading = true;
    });

    try {
      final profileData = await ProfileService.getCachedProfile();
      if (profileData != null) {
        setState(() {
          username = profileData['username'];
          email = profileData['email'];
          profilePictureUrl = profileData['profile_picture'];
        });
      } else {
        // Optionally, you can fetch profile data from the server if not found in cache
        final profile = await ProfileService.fetchProfile();
        setState(() {
          username = profile['username'];
          email = profile['email'];
          profilePictureUrl = profile['profile_picture'];
        });
      }
    } catch (e) {
      Fluttertoast.showToast(msg: 'Error loading profile: $e');
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  // Method to pick and update profile picture
  Future<void> _updateProfilePicture() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        isLoading = true;
      });
      try {
        File imageFile = File(pickedFile.path);
        await ProfileService.updateProfilePicture(imageFile);
        // After the update, reload the profile to reflect changes
        await _loadProfileData();
        Fluttertoast.showToast(msg: 'Profile picture updated');
      } catch (e) {
        Fluttertoast.showToast(msg: 'Error updating profile picture: $e');
      } finally {
        setState(() {
          isLoading = false;
        });
      }
    }
  }

  // Method to update user profile
  Future<void> _updateProfile() async {
    if (username == null || email == null) {
      Fluttertoast.showToast(msg: 'Please fill in all fields');
      return;
    }

    setState(() {
      isLoading = true;
    });

    try {
      final updatedProfileData = {
        'username': username,
        'email': email,
      };

      await ProfileService.updateProfile(updatedProfileData);
      Fluttertoast.showToast(msg: 'Profile updated successfully');
      await _loadProfileData(); // Reload profile after update
    } catch (e) {
      Fluttertoast.showToast(msg: 'Error updating profile: $e');
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Profile'),
      ),
      body: isLoading
          ? Center(child: CircularProgressIndicator())
          : Padding(
              padding: EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Profile Picture
                  Center(
                    child: GestureDetector(
                      onTap: _updateProfilePicture,
                      child: CircleAvatar(
                        radius: 60,
                        backgroundImage: profilePictureUrl != null
                            ? NetworkImage(profilePictureUrl!)
                            : AssetImage('assets/default_profile.png')
                                as ImageProvider,
                      ),
                    ),
                  ),
                  SizedBox(height: 20),
                  // Username Field
                  TextField(
                    decoration: InputDecoration(
                      labelText: 'Username',
                      border: OutlineInputBorder(),
                    ),
                    onChanged: (value) {
                      setState(() {
                        username = value;
                      });
                    },
                    controller: TextEditingController(text: username),
                  ),
                  SizedBox(height: 10),
                  // Email Field
                  TextField(
                    decoration: InputDecoration(
                      labelText: 'Email',
                      border: OutlineInputBorder(),
                    ),
                    onChanged: (value) {
                      setState(() {
                        email = value;
                      });
                    },
                    controller: TextEditingController(text: email),
                  ),
                  SizedBox(height: 20),
                  // Update Profile Button
                  Center(
                    child: ElevatedButton(
                      onPressed: _updateProfile,
                      child: Text('Update Profile'),
                    ),
                  ),
                ],
              ),
            ),
    );
  }
}
