import 'package:flutter/material.dart';

class JournalSearchBar extends StatelessWidget {
  final TextEditingController searchController;
  final Function(String) onSearch;
  final bool isDarkMode;
  final Function(String) onSortChanged;
  final String currentSort;

  const JournalSearchBar({
    super.key,
    required this.searchController,
    required this.onSearch,
    required this.isDarkMode,
    required this.onSortChanged,
    required this.currentSort,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: Container(
            height: 40,
            decoration: BoxDecoration(
              color: isDarkMode ? Colors.grey[800] : Colors.white,
              borderRadius: BorderRadius.circular(8),
              boxShadow: [
                BoxShadow(
                  color: Colors.black12,
                  blurRadius: 4,
                  offset: Offset(0, 2),
                ),
              ],
            ),
            child: TextField(
              controller: searchController,
              onChanged: onSearch,
              style: TextStyle(fontSize: 14),
              decoration: InputDecoration(
                hintText: 'Search journals...',
                prefixIcon: Icon(Icons.search, size: 20),
                border: InputBorder.none,
                contentPadding: EdgeInsets.symmetric(
                  horizontal: 8,
                  vertical: 4,
                ),
              ),
            ),
          ),
        ),
        SizedBox(width: 8),
        Container(
          height: 40,
          decoration: BoxDecoration(
            color: isDarkMode ? Colors.grey[800] : Colors.white,
            borderRadius: BorderRadius.circular(8),
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
              value: currentSort,
              isDense: true,
              padding: EdgeInsets.symmetric(horizontal: 8),
              icon: Icon(Icons.sort, size: 20),
              items: [
                DropdownMenuItem(
                  value: 'newest',
                  child: Text('Newest', style: TextStyle(fontSize: 13)),
                ),
                DropdownMenuItem(
                  value: 'oldest',
                  child: Text('Oldest', style: TextStyle(fontSize: 13)),
                ),
                DropdownMenuItem(
                  value: 'title',
                  child: Text('Title', style: TextStyle(fontSize: 13)),
                ),
              ],
              onChanged: (String? value) {
                if (value != null) {
                  onSortChanged(value);
                }
              },
            ),
          ),
        ),
      ],
    );
  }
}
