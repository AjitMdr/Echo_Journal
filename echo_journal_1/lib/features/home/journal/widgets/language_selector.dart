import 'package:flutter/material.dart';

class LanguageSelectorWidget extends StatefulWidget {
  final VoidCallback onRefresh;
  final String selectedLanguage;
  final Function(String) onLanguageChanged;

  const LanguageSelectorWidget({
    super.key,
    required this.onRefresh,
    required this.selectedLanguage,
    required this.onLanguageChanged,
  });

  @override
  State<LanguageSelectorWidget> createState() => _LanguageSelectorWidgetState();
}

class _LanguageSelectorWidgetState extends State<LanguageSelectorWidget> {
  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Row(
          children: [
            Text(
              'Language:',
              style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
            ),
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
                  value: widget.selectedLanguage,
                  items: [
                    DropdownMenuItem(
                      value: 'en',
                      child: Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 8.0),
                        child: Text('English'),
                      ),
                    ),
                    DropdownMenuItem(
                      value: 'ne',
                      child: Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 8.0),
                        child: Text('Nepali'),
                      ),
                    ),
                  ],
                  onChanged: (String? newValue) {
                    if (newValue != null) {
                      widget.onLanguageChanged(newValue);
                    }
                  },
                ),
              ),
            ),
          ],
        ),
        IconButton(
          icon: Icon(Icons.refresh, color: Colors.blue),
          onPressed: widget.onRefresh,
          tooltip: 'Refresh Journals',
        ),
      ],
    );
  }
}
