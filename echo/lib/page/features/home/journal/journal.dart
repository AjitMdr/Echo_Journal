import 'package:flutter/material.dart';
import 'package:intl/intl.dart' show DateFormat;
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:speech_to_text/speech_recognition_result.dart';
import 'package:translator/translator.dart';
import '../../../../services/journal/journal_service.dart';
import '../../../core/constants/curved_pattern.dart';
import '../../../../themes/theme_provider.dart';

class JournalPage extends StatefulWidget {
  const JournalPage({super.key});

  @override
  JournalPageState createState() => JournalPageState();
}

class JournalPageState extends State<JournalPage> {
  String _selectedLanguage = 'en';
  final TextEditingController _titleController = TextEditingController();
  final TextEditingController _contentController = TextEditingController();
  List<Map<String, dynamic>> _journals = [];
  bool _isLoading = false;
  int? _editingJournalId;

  // Voice recognition
  final stt.SpeechToText _speech = stt.SpeechToText();
  bool _isListening = false;
  String _currentField = 'content'; // 'title' or 'content'
  final translator = GoogleTranslator();

  final List<String> _languages = ['en', 'ne'];
  final Map<String, String> _languageNames = {
    'en': 'English',
    'ne': 'Nepali',
  };

  Future<void> _loadLanguage() async {
    try {
      SharedPreferences prefs = await SharedPreferences.getInstance();
      String? language = prefs.getString('language');
      if (mounted) {
        setState(() {
          _selectedLanguage = language ?? 'en';
        });
      }
    } catch (e) {
      print('Error loading language preference: $e');
    }
  }

  Future<void> _saveLanguage(String language) async {
    try {
      SharedPreferences prefs = await SharedPreferences.getInstance();
      await prefs.setString('language', language);
      _fetchJournals();
    } catch (e) {
      print('Error saving language preference: $e');
    }
  }

  Future<void> _fetchJournals() async {
    try {
      if (mounted) {
        setState(() {
          _isLoading = true;
        });
      }
      var journals = await JournalService.fetchJournals();
      if (mounted) {
        setState(() {
          _journals = journals;
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
        _showSnackBar('Failed to load journals', isError: true);
      }
    }
  }

  Future<void> _createJournal() async {
    String title = _titleController.text.trim();
    String content = _contentController.text.trim();

    if (title.isEmpty || content.isEmpty) {
      _showSnackBar('Title and content cannot be empty', isError: true);
      return;
    }

    try {
      if (mounted) {
        setState(() {
          _isLoading = true;
        });
      }
      bool success = await JournalService.createJournal(title, content);
      if (mounted) {
        setState(() {
          _isLoading = false;
        });

        if (success) {
          _titleController.clear();
          _contentController.clear();
          _showSnackBar('Journal saved successfully');
          _fetchJournals();
        } else {
          _showSnackBar('Failed to save journal', isError: true);
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
        _showSnackBar('Error: ${e.toString()}', isError: true);
      }
    }
  }

  Future<void> _deleteJournal(int journalId) async {
    try {
      if (mounted) {
        setState(() {
          _isLoading = true;
        });
      }
      bool success = await JournalService.deleteJournal(journalId);
      if (mounted) {
        setState(() {
          _isLoading = false;
        });

        if (success) {
          _showSnackBar('Journal deleted successfully');
          _fetchJournals();
        } else {
          _showSnackBar('Failed to delete journal', isError: true);
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
        _showSnackBar('Error: ${e.toString()}', isError: true);
      }
    }
  }

  Future<void> _updateJournal() async {
    if (_editingJournalId == null) {
      _showSnackBar('No journal selected for update', isError: true);
      return;
    }

    String title = _titleController.text.trim();
    String content = _contentController.text.trim();

    if (title.isEmpty || content.isEmpty) {
      _showSnackBar('Title and content cannot be empty', isError: true);
      return;
    }

    try {
      if (mounted) {
        setState(() {
          _isLoading = true;
        });
      }
      bool success = await JournalService.updateJournal(
          _editingJournalId!, title, content);
      if (mounted) {
        setState(() {
          _isLoading = false;
          _editingJournalId = null;
        });

        if (success) {
          _titleController.clear();
          _contentController.clear();
          _showSnackBar('Journal updated successfully');
          _fetchJournals();
        } else {
          _showSnackBar('Failed to update journal', isError: true);
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _editingJournalId = null;
        });
        _showSnackBar('Error: ${e.toString()}', isError: true);
      }
    }
  }

  void _prepareForEdit(Map<String, dynamic> journal) {
    if (mounted) {
      setState(() {
        _editingJournalId = journal['id'];
        _titleController.text = journal['title'] ?? '';
        _contentController.text = journal['content'] ?? '';
      });
      _showSnackBar('Editing journal: ${journal['title']}');
    }
  }

  void _showSnackBar(String message, {bool isError = false}) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(message),
          backgroundColor: isError ? Colors.red : Colors.green,
          duration: Duration(seconds: 3),
        ),
      );
    }
  }

  void _cancelEditing() {
    if (mounted) {
      setState(() {
        _editingJournalId = null;
        _titleController.clear();
        _contentController.clear();
      });
    }
  }

  // Initialize speech recognition
  Future<void> _initSpeech() async {
    bool available = await _speech.initialize(
      onStatus: _onSpeechStatus,
      onError: (error) {
        if (mounted) {
          _showSnackBar('Error: $error', isError: true);
        }
      },
    );
    if (!available && mounted) {
      _showSnackBar('Speech recognition not available on this device',
          isError: true);
    }
  }

  // Listen for speech
  Future<void> _startListening(String field) async {
    if (mounted) {
      setState(() {
        _currentField = field;
      });
    }

    if (!_speech.isAvailable) {
      await _initSpeech();
    }

    // Start listening in the appropriate language
    String localeId = _selectedLanguage == 'en' ? 'en_US' : 'ne_NP';

    await _speech.listen(
      onResult: _onSpeechResult,
      localeId: localeId,
    );

    if (mounted) {
      setState(() {
        _isListening = true;
      });
      _showSnackBar(
          'Listening for ${_languageNames[_selectedLanguage]} speech...');
    }
  }

  // Stop listening
  void _stopListening() {
    _speech.stop();
    if (mounted) {
      setState(() {
        _isListening = false;
      });
    }
  }

  // Handle speech recognition status changes
  void _onSpeechStatus(String status) {
    if (status == 'notListening' && mounted) {
      setState(() {
        _isListening = false;
      });
    }
  }

  // Handle speech recognition results
  void _onSpeechResult(SpeechRecognitionResult result) async {
    if (result.finalResult) {
      String recognizedText = result.recognizedWords;

      // If the recognized text is not in the desired language, translate it
      if (!_speech.isListening) {
        String targetLanguage = _selectedLanguage;
        String recognizedLanguage = await _detectLanguage(recognizedText);

        if (recognizedLanguage != targetLanguage) {
          try {
            var translation = await translator.translate(
              recognizedText,
              from: recognizedLanguage,
              to: targetLanguage,
            );
            recognizedText = translation.text;
          } catch (e) {
            print('Translation error: $e');
            // If translation fails, use the original text
          }
        }

        // Check if widget is still mounted before updating state
        if (mounted) {
          setState(() {
            if (_currentField == 'title') {
              _titleController.text = _titleController.text.isEmpty
                  ? recognizedText
                  : '${_titleController.text} $recognizedText';
            } else {
              _contentController.text = _contentController.text.isEmpty
                  ? recognizedText
                  : '${_contentController.text} $recognizedText';
            }
          });
        }
      }
    }
  }

  // Detect language of the text
  Future<String> _detectLanguage(String text) async {
    try {
      var detected = await translator.translate(text);
      return detected.sourceLanguage.code;
    } catch (e) {
      print('Language detection error: $e');
      // Default to English if detection fails
      return 'en';
    }
  }

  @override
  void initState() {
    super.initState();
    _loadLanguage();
    _fetchJournals();
    _initSpeech();
  }

  @override
  void dispose() {
    _titleController.dispose();
    _contentController.dispose();
    _speech.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final isDarkMode = themeProvider.isDarkMode;
    return Scaffold(
      body: Stack(
        children: [
          // Background pattern
          CustomPaint(
            painter: CurvedPatternPainter(),
            size: MediaQuery.of(context).size,
          ),

          // Main content
          _isLoading
              ? Center(child: CircularProgressIndicator())
              : Padding(
                  padding: const EdgeInsets.fromLTRB(12.0, 16.0, 12.0, 12.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Language selector and refresh button
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Row(
                            children: [
                              Text('Language:',
                                  style: TextStyle(
                                      fontSize: 14,
                                      fontWeight: FontWeight.bold)),
                              SizedBox(width: 8),
                              Container(
                                height: 36,
                                decoration: BoxDecoration(
                                  borderRadius: BorderRadius.circular(8),
                                  color: Colors.white,
                                  boxShadow: [
                                    BoxShadow(
                                      color: Colors.black12,
                                      blurRadius: 4,
                                      offset: Offset(0, 2),
                                    ),
                                  ],
                                ),
                                child: DropdownButtonHideUnderline(
                                  child: DropdownButton<String>(
                                    value: _selectedLanguage,
                                    items: _languages.map((String lang) {
                                      return DropdownMenuItem<String>(
                                        value: lang,
                                        child: Padding(
                                          padding: const EdgeInsets.symmetric(
                                              horizontal: 8.0),
                                          child: Text(
                                              _languageNames[lang] ?? lang),
                                        ),
                                      );
                                    }).toList(),
                                    onChanged: (String? newValue) {
                                      if (newValue != null &&
                                          newValue != _selectedLanguage) {
                                        if (mounted) {
                                          setState(() {
                                            _selectedLanguage = newValue;
                                          });
                                        }
                                        _saveLanguage(newValue);
                                      }
                                    },
                                  ),
                                ),
                              ),
                            ],
                          ),
                          IconButton(
                            icon: Icon(Icons.refresh, color: Colors.blue),
                            onPressed: _fetchJournals,
                            tooltip: 'Refresh Journals',
                          ),
                        ],
                      ),
                      SizedBox(height: 10),

                      // Journal input form
                      Card(
                        elevation: 3,
                        color: Colors.white,
                        margin: EdgeInsets.symmetric(vertical: 8),
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8)),
                        child: Padding(
                          padding: const EdgeInsets.all(12.0),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.stretch,
                            children: [
                              Row(
                                children: [
                                  Icon(
                                      _editingJournalId == null
                                          ? Icons.create
                                          : Icons.edit,
                                      size: 20,
                                      color: Colors.blueGrey),
                                  SizedBox(width: 8),
                                  Text(
                                    _editingJournalId == null
                                        ? 'Create New Journal'
                                        : 'Edit Journal',
                                    style: TextStyle(
                                        fontSize: 16,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.blueGrey),
                                  ),
                                ],
                              ),
                              Divider(height: 20),

                              // Title field
                              Row(
                                children: [
                                  Expanded(
                                    child: TextField(
                                      controller: _titleController,
                                      style: TextStyle(fontSize: 14),
                                      decoration: InputDecoration(
                                        labelText: 'Title',
                                        border: OutlineInputBorder(
                                          borderRadius:
                                              BorderRadius.circular(8),
                                          borderSide: BorderSide(
                                              width: 1, color: Colors.blueGrey),
                                        ),
                                        contentPadding: EdgeInsets.symmetric(
                                            horizontal: 10, vertical: 8),
                                        prefixIcon: Icon(Icons.title, size: 18),
                                        isDense: true,
                                      ),
                                    ),
                                  ),
                                  IconButton(
                                    icon: Icon(
                                      _isListening && _currentField == 'title'
                                          ? Icons.mic
                                          : Icons.mic_none,
                                      color: _isListening &&
                                              _currentField == 'title'
                                          ? Colors.red
                                          : Colors.grey,
                                      size: 20,
                                    ),
                                    onPressed:
                                        _isListening && _currentField == 'title'
                                            ? _stopListening
                                            : () => _startListening('title'),
                                    tooltip:
                                        _isListening && _currentField == 'title'
                                            ? 'Stop Recording'
                                            : 'Record Title',
                                    padding: EdgeInsets.all(8),
                                    constraints: BoxConstraints(),
                                  ),
                                ],
                              ),
                              SizedBox(height: 10),

                              // Content field
                              Row(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Expanded(
                                    child: TextField(
                                      controller: _contentController,
                                      style: TextStyle(fontSize: 14),
                                      decoration: InputDecoration(
                                        labelText: 'Content',
                                        border: OutlineInputBorder(
                                          borderRadius:
                                              BorderRadius.circular(8),
                                          borderSide: BorderSide(
                                              width: 1, color: Colors.blueGrey),
                                        ),
                                        contentPadding: EdgeInsets.symmetric(
                                            horizontal: 10, vertical: 8),
                                        prefixIcon:
                                            Icon(Icons.edit_note, size: 18),
                                        alignLabelWithHint: true,
                                        isDense: true,
                                      ),
                                      maxLines: 4,
                                    ),
                                  ),
                                  IconButton(
                                    icon: Icon(
                                      _isListening && _currentField == 'content'
                                          ? Icons.mic
                                          : Icons.mic_none,
                                      color: _isListening &&
                                              _currentField == 'content'
                                          ? Colors.red
                                          : Colors.grey,
                                      size: 20,
                                    ),
                                    onPressed: _isListening &&
                                            _currentField == 'content'
                                        ? _stopListening
                                        : () => _startListening('content'),
                                    tooltip: _isListening &&
                                            _currentField == 'content'
                                        ? 'Stop Recording'
                                        : 'Record Content',
                                    padding: EdgeInsets.all(8),
                                    constraints: BoxConstraints(),
                                  ),
                                ],
                              ),
                              SizedBox(height: 12),

                              // Action buttons
                              Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceBetween,
                                children: [
                                  if (_editingJournalId != null)
                                    ElevatedButton.icon(
                                      onPressed: _cancelEditing,
                                      icon: Icon(Icons.cancel, size: 16),
                                      label: Text('Cancel',
                                          style: TextStyle(fontSize: 13)),
                                      style: ElevatedButton.styleFrom(
                                        backgroundColor: Colors.grey,
                                        padding: EdgeInsets.symmetric(
                                            horizontal: 12, vertical: 8),
                                        minimumSize: Size(80, 36),
                                      ),
                                    ),
                                  Expanded(
                                    child: Padding(
                                      padding: EdgeInsets.only(
                                          left: _editingJournalId != null
                                              ? 8.0
                                              : 0),
                                      child: ElevatedButton.icon(
                                        onPressed: _editingJournalId == null
                                            ? _createJournal
                                            : _updateJournal,
                                        icon: Icon(
                                            _editingJournalId == null
                                                ? Icons.save
                                                : Icons.update,
                                            size: 16),
                                        label: Text(
                                          _editingJournalId == null
                                              ? 'Save Journal'
                                              : 'Update Journal',
                                          style: TextStyle(fontSize: 13),
                                        ),
                                        style: ElevatedButton.styleFrom(
                                          backgroundColor:
                                              _editingJournalId == null
                                                  ? Colors.blue
                                                  : Colors.green,
                                          padding: EdgeInsets.symmetric(
                                              horizontal: 12, vertical: 8),
                                          minimumSize: Size(100, 36),
                                        ),
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ),

                      SizedBox(height: 10),

                      // Journal list header
                      Padding(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 4.0, vertical: 4.0),
                        child: Row(
                          children: [
                            Icon(Icons.book, size: 16, color: Colors.blueGrey),
                            SizedBox(width: 8),
                            Text(
                              'My Journals',
                              style: TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                                color: Colors.blueGrey,
                              ),
                            ),
                            SizedBox(width: 8),
                            Expanded(
                              child: Divider(
                                thickness: 2,
                                color: Colors.blueGrey.withOpacity(0.3),
                              ),
                            ),
                          ],
                        ),
                      ),

                      // Journal list
                      Expanded(
                        child: _journals.isEmpty
                            ? Center(
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Icon(Icons.book,
                                        size: 48, color: Colors.grey),
                                    SizedBox(height: 12),
                                    Text(
                                      'No journals found',
                                      style: TextStyle(
                                          fontSize: 16, color: Colors.grey),
                                    ),
                                    SizedBox(height: 4),
                                    Text(
                                      'Create your first journal entry above',
                                      style: TextStyle(
                                          color: Colors.grey, fontSize: 13),
                                    ),
                                  ],
                                ),
                              )
                            : ListView.builder(
                                itemCount: _journals.length,
                                itemBuilder: (context, index) {
                                  var journal = _journals[index];

                                  String formattedDate = journal['date'] !=
                                              null &&
                                          journal['date'].isNotEmpty
                                      ? DateFormat('yyyy-MM-dd – kk:mm').format(
                                          DateTime.parse(journal['date'])
                                              .toLocal())
                                      : 'Created: Unknown';

                                  return Card(
                                    color: isDarkMode
                                        ? Colors.grey[800]
                                        : Colors.white,
                                    margin: EdgeInsets.symmetric(
                                        vertical: 6, horizontal: 2),
                                    elevation: 2,
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(8),
                                    ),
                                    child: ListTile(
                                      contentPadding: EdgeInsets.symmetric(
                                          vertical: 8, horizontal: 12),
                                      title: Text(
                                        formattedDate,
                                        style: TextStyle(
                                          fontSize: 14,
                                          fontStyle: FontStyle.italic,
                                          color: isDarkMode
                                              ? Colors.white
                                              : Colors.black,
                                        ),
                                      ),
                                      subtitle: Column(
                                        crossAxisAlignment:
                                            CrossAxisAlignment.start,
                                        children: [
                                          SizedBox(height: 3),
                                          Text(
                                            journal['title'] ?? 'No Title',
                                            style: TextStyle(
                                              fontSize: 16,
                                              fontWeight: FontWeight.bold,
                                              color: isDarkMode
                                                  ? Colors.white
                                                  : Colors.black,
                                            ),
                                          ),
                                          SizedBox(height: 3),
                                          Text(
                                            journal['content'] ?? 'No Content',
                                            maxLines: 2,
                                            overflow: TextOverflow.ellipsis,
                                            style: TextStyle(
                                              fontSize: 14,
                                              color: isDarkMode
                                                  ? Colors.white
                                                  : Colors.black,
                                            ),
                                          ),
                                          SizedBox(height: 3),
                                          // Display created_at if it's available
                                          if (journal['created_at'] != null)
                                            Text(
                                              'Created: ${journal['created_at']}',
                                              style: TextStyle(
                                                fontSize: 11,
                                                fontStyle: FontStyle.italic,
                                                color: isDarkMode
                                                    ? Colors.white
                                                    : Colors.black,
                                              ),
                                            ),
                                        ],
                                      ),
                                      trailing: Row(
                                        mainAxisSize: MainAxisSize.min,
                                        children: [
                                          IconButton(
                                            icon: Icon(Icons.edit,
                                                color: Colors.blue, size: 18),
                                            onPressed: () =>
                                                _prepareForEdit(journal),
                                            tooltip: 'Edit Journal',
                                            padding: EdgeInsets.all(6),
                                            constraints: BoxConstraints(),
                                          ),
                                          SizedBox(width: 8),
                                          IconButton(
                                            icon: Icon(Icons.delete,
                                                color: Colors.red, size: 18),
                                            onPressed: () =>
                                                _showDeleteConfirmation(
                                                    journal),
                                            tooltip: 'Delete Journal',
                                            padding: EdgeInsets.all(6),
                                            constraints: BoxConstraints(),
                                          ),
                                        ],
                                      ),
                                    ),
                                  );
                                },
                              ),
                      ),
                    ],
                  ),
                ),
        ],
      ),
    );
  }

  Future<void> _showDeleteConfirmation(Map<String, dynamic> journal) async {
    if (!mounted) return;

    return showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Confirm Delete', style: TextStyle(fontSize: 16)),
          content: Text(
              'Are you sure you want to delete "${journal['title']}"?',
              style: TextStyle(fontSize: 14)),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: Text('Cancel'),
            ),
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
                _deleteJournal(journal['id']);
              },
              style: TextButton.styleFrom(foregroundColor: Colors.red),
              child: Text('Delete'),
            ),
          ],
        );
      },
    );
  }
}
