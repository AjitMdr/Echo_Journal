from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from django.shortcuts import get_object_or_404

from .models import Journal
from .serializers import JournalSerializer, JournalCreateUpdateSerializer

@api_view(['GET', 'POST'])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([IsAuthenticated])
def journal_view(request):
    """Retrieve all journal entries or create a new one."""
    if request.method == 'GET':
        journals = Journal.objects.filter(user=request.user).order_by('-date')
        serializer = JournalSerializer(journals, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = JournalCreateUpdateSerializer(data=request.data)
        if serializer.is_valid():
            journal = serializer.save(user=request.user)
            return Response(JournalSerializer(journal).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'PUT', 'DELETE'])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([IsAuthenticated])
def journal_detail_view(request, journal_id):
    """Retrieve, update, or delete a specific journal entry."""
    journal = get_object_or_404(Journal, id=journal_id, user=request.user)

    if request.method == 'GET':
        serializer = JournalSerializer(journal)
        return Response(serializer.data)

    elif request.method == 'PUT':
        serializer = JournalCreateUpdateSerializer(journal, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(JournalSerializer(serializer.instance).data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        journal.delete()
        return Response({'message': 'Journal entry deleted successfully'}, status=status.HTTP_204_NO_CONTENT)
