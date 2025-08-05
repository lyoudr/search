import os.path
from pytz import timezone 
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from datetime import timedelta 



SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_calendar_service():
    creds = None 
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json')
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    service = build('calendar', 'v3', credentials=creds)
    return service


def check_availability(date_str: str) -> str:
    """
    date_str format: '2025-08-03'
    """
    service = get_calendar_service()

    # Define time range (e.g., 8 AM to 6 PM)
    from dateutil.parser import parse
    tz = timezone('Asia/Taipei')
    start_time = parse(date_str + " 08:00:00")
    start_time = start_time.astimezone(tz)
    end_time = parse(date_str + " 18:00:00")
    end_time = end_time.astimezone(tz)

    events_result = service.events().list(
        calendarId='lyoudr@gmail.com',
        timeMin=start_time.isoformat(),
        timeMax=end_time.isoformat(),
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    events = events_result.get('items', [])

    if not events:
        return f"{date_str} is completely free!"
    
    # Build availability list
    availability = []
    last_end = start_time 

    from dateutil.parser import parse
    for event in events:
        start = parse(event['start']['dateTime'])
        if start > last_end:
            availability.append(f"{last_end.time()} - {start.time()}")
        last_end = parse(event['end']['dateTime'])

    # Check for remaining free time at the end
    if last_end < end_time:
        availability.append(f"{last_end.time()} - {end_time.time()}")
    
    if availability:
        return f"Available slots on {date_str}: " + ", ".join(availability)
    else:
        return f"No availability on {date_str}"


def check_availability_tool(input: str) -> str:
    """
    Expects input in 'YYYY-MM-DD' format
    """
    return check_availability(input)


# 🧰 2. Tool: BookCalendar — Insert a Google Calendar Event
def book_calendar_tool(input: str) -> str:
    """
    Input format: "YYYY-MM-DD HH:MM, Service name, Customer name"
    """
    try:
        parts = [s.strip() for s in input.split(',')]
        if len(parts) != 3:
            return "❌ Input format error: Please provide 'YYYY-MM-DD HH:MM, Service name, Customer name'"
        date_str, service, customer = parts
        from dateutil.parser import parse
        start_time = parse(date_str)
        end_time = start_time + timedelta(hours=1)

        service_obj = get_calendar_service()

        event = {
            'summary': f"{service} - {customer}",
            'description': f"{service} appointment with {customer}",
            'start': {'dateTime': start_time.isoformat(), 'timeZone': 'Asia/Taipei'},
            'end': {'dateTime': end_time.isoformat(), 'timeZone': 'Asia/Taipei'}
        }

        event_result = service_obj.events().insert(calendarId='primary', body=event).execute()
        return f"✅ Booked: {event_result['summary']} on {start_time.strftime('%Y-%m-%d %H:%M')}"
    except Exception as e:
        return f"❌ Failed to book calendar: {str(e)}"
    

