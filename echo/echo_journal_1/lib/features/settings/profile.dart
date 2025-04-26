import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:echo_journal1/features/widgets/streak_badge_widget.dart';
import 'package:provider/provider.dart';
import 'package:echo_journal1/core/configs/theme/theme-provider.dart';
import 'package:echo_journal1/services/profile_service.dart';

class ProfilePage extends StatefulWidget {
  const ProfilePage({super.key});

  @override
  _ProfilePageState createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  String? username;
  String? email;
  String? profilePictureUrl;
  bool isLoading = false;
  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _loadProfileData();
  }

  Future<void> _loadProfileData() async {
    setState(() {
      isLoading = true;
    });

    try {
      final profileData = await ProfileService.getProfile();
      setState(() {
        username = profileData['username'];
        email = profileData['email'];
        profilePictureUrl = profileData['profile_picture'];
      });
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error loading profile: $e')),
      );
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  Future<void> _updateProfilePicture() async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: ImageSource.gallery,
        imageQuality: 80,
      );

      if (pickedFile == null) return;

      setState(() {
        isLoading = true;
      });

      File imageFile = File(pickedFile.path);

      try {
        final response = await ProfileService.updateProfilePicture(imageFile);
        setState(() {
          profilePictureUrl = response['profile_picture'];
        });

        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Profile picture updated successfully'),
            backgroundColor: Colors.green,
          ),
        );
      } catch (e) {
        String errorMessage = e.toString();
        if (errorMessage.contains('Unauthorized') ||
            errorMessage.contains('Session expired')) {
          // Handle authentication error
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Please log in again to continue'),
              backgroundColor: Colors.red,
              duration: Duration(seconds: 5),
            ),
          );
          // TODO: Navigate to login screen or handle re-authentication
          // Navigator.of(context).pushReplacementNamed('/login');
        } else {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Error updating profile picture: ${e.toString()}'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error selecting image: ${e.toString()}'),
          backgroundColor: Colors.red,
        ),
      );
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  Future<void> _updateProfile() async {
    if (username == null || email == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Please fill in all fields')),
      );
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
      await _loadProfileData();

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Profile updated successfully')),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error updating profile: $e')),
      );
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final isDarkMode = themeProvider.isDarkMode;

    return Scaffold(
      appBar: AppBar(title: Text('Profile')),
      body: isLoading
          ? Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              child: Padding(
                padding: EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Center(
                      child: Stack(
                        children: [
                          GestureDetector(
                            onTap: _updateProfilePicture,
                            child: CircleAvatar(
                              radius: 60,
                              backgroundImage: profilePictureUrl != null
                                  ? NetworkImage(profilePictureUrl!)
                                  : AssetImage('assets/default_profile.png')
                                      as ImageProvider,
                            ),
                          ),
                          Positioned(
                            bottom: 0,
                            right: 0,
                            child: Container(
                              padding: EdgeInsets.all(4),
                              decoration: BoxDecoration(
                                color: Theme.of(context).primaryColor,
                                shape: BoxShape.circle,
                              ),
                              child: Icon(
                                Icons.camera_alt,
                                color: Colors.white,
                                size: 20,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    SizedBox(height: 20),
                    StreakBadgeWidget(isDarkMode: isDarkMode),
                    SizedBox(height: 20),
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
                    Center(
                      child: ElevatedButton(
                        onPressed: _updateProfile,
                        child: Text('Update Profile'),
                      ),
                    ),
                  ],
                ),
              ),
            ),
    );
  }
}
