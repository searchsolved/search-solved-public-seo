"""
Schema Markup Generator - Streamlit App
Generate valid JSON-LD schema markup for common schema types.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import json
from datetime import datetime, date

st.set_page_config(
    page_title="Schema Markup Generator",
    page_icon="🏷️",
    layout="wide"
)

st.title("🏷️ Schema Markup Generator")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Generates structured data markup
    - Creates JSON-LD for common schemas
    - Validates generated markup

    **How to use:**
    1. Select schema type
    2. Fill in required fields
    3. Generate JSON-LD markup
    4. Copy/download the code

    **Best for:**
    - Structured data implementation
    - Rich result optimization
    - Schema creation at scale
    """)
st.markdown("Generate valid JSON-LD structured data for your pages.")


def generate_faq_schema(faqs):
    """Generate FAQPage schema."""
    schema = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": []
    }

    for faq in faqs:
        if faq['question'] and faq['answer']:
            schema["mainEntity"].append({
                "@type": "Question",
                "name": faq['question'],
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": faq['answer']
                }
            })

    return schema


def generate_howto_schema(data):
    """Generate HowTo schema."""
    schema = {
        "@context": "https://schema.org",
        "@type": "HowTo",
        "name": data['name'],
        "description": data['description'],
        "step": []
    }

    if data.get('total_time'):
        schema["totalTime"] = data['total_time']

    if data.get('estimated_cost'):
        schema["estimatedCost"] = {
            "@type": "MonetaryAmount",
            "currency": data.get('currency', 'USD'),
            "value": data['estimated_cost']
        }

    if data.get('image'):
        schema["image"] = data['image']

    for i, step in enumerate(data['steps'], 1):
        if step['text']:
            step_data = {
                "@type": "HowToStep",
                "position": i,
                "name": step.get('name', f"Step {i}"),
                "text": step['text']
            }
            if step.get('image'):
                step_data["image"] = step['image']
            schema["step"].append(step_data)

    return schema


def generate_article_schema(data):
    """Generate Article schema."""
    schema = {
        "@context": "https://schema.org",
        "@type": data.get('article_type', 'Article'),
        "headline": data['headline'],
        "description": data['description'],
        "author": {
            "@type": data.get('author_type', 'Person'),
            "name": data['author_name']
        },
        "datePublished": data['date_published'],
        "publisher": {
            "@type": "Organization",
            "name": data['publisher_name']
        }
    }

    if data.get('date_modified'):
        schema["dateModified"] = data['date_modified']

    if data.get('image'):
        schema["image"] = data['image']

    if data.get('publisher_logo'):
        schema["publisher"]["logo"] = {
            "@type": "ImageObject",
            "url": data['publisher_logo']
        }

    if data.get('author_url'):
        schema["author"]["url"] = data['author_url']

    return schema


def generate_product_schema(data):
    """Generate Product schema."""
    schema = {
        "@context": "https://schema.org",
        "@type": "Product",
        "name": data['name'],
        "description": data['description']
    }

    if data.get('image'):
        schema["image"] = data['image']

    if data.get('brand'):
        schema["brand"] = {
            "@type": "Brand",
            "name": data['brand']
        }

    if data.get('sku'):
        schema["sku"] = data['sku']

    if data.get('mpn'):
        schema["mpn"] = data['mpn']

    if data.get('gtin'):
        schema["gtin"] = data['gtin']

    # Offer
    if data.get('price'):
        schema["offers"] = {
            "@type": "Offer",
            "price": data['price'],
            "priceCurrency": data.get('currency', 'USD'),
            "availability": f"https://schema.org/{data.get('availability', 'InStock')}",
            "url": data.get('url', '')
        }

        if data.get('price_valid_until'):
            schema["offers"]["priceValidUntil"] = data['price_valid_until']

    # Reviews
    if data.get('rating_value') and data.get('review_count'):
        schema["aggregateRating"] = {
            "@type": "AggregateRating",
            "ratingValue": data['rating_value'],
            "reviewCount": data['review_count'],
            "bestRating": data.get('best_rating', '5'),
            "worstRating": data.get('worst_rating', '1')
        }

    return schema


def generate_localbusiness_schema(data):
    """Generate LocalBusiness schema."""
    schema = {
        "@context": "https://schema.org",
        "@type": data.get('business_type', 'LocalBusiness'),
        "name": data['name'],
        "address": {
            "@type": "PostalAddress",
            "streetAddress": data['street_address'],
            "addressLocality": data['city'],
            "addressRegion": data['state'],
            "postalCode": data['postal_code'],
            "addressCountry": data['country']
        }
    }

    if data.get('description'):
        schema["description"] = data['description']

    if data.get('phone'):
        schema["telephone"] = data['phone']

    if data.get('url'):
        schema["url"] = data['url']

    if data.get('image'):
        schema["image"] = data['image']

    if data.get('price_range'):
        schema["priceRange"] = data['price_range']

    if data.get('latitude') and data.get('longitude'):
        schema["geo"] = {
            "@type": "GeoCoordinates",
            "latitude": data['latitude'],
            "longitude": data['longitude']
        }

    # Opening hours
    if data.get('opening_hours'):
        schema["openingHoursSpecification"] = []
        for hours in data['opening_hours']:
            if hours['days'] and hours['open'] and hours['close']:
                schema["openingHoursSpecification"].append({
                    "@type": "OpeningHoursSpecification",
                    "dayOfWeek": hours['days'],
                    "opens": hours['open'],
                    "closes": hours['close']
                })

    return schema


def generate_breadcrumb_schema(breadcrumbs):
    """Generate BreadcrumbList schema."""
    schema = {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": []
    }

    for i, crumb in enumerate(breadcrumbs, 1):
        if crumb['name']:
            item = {
                "@type": "ListItem",
                "position": i,
                "name": crumb['name']
            }
            if crumb.get('url'):
                item["item"] = crumb['url']
            schema["itemListElement"].append(item)

    return schema


def generate_video_schema(data):
    """Generate VideoObject schema."""
    schema = {
        "@context": "https://schema.org",
        "@type": "VideoObject",
        "name": data['name'],
        "description": data['description'],
        "thumbnailUrl": data['thumbnail_url'],
        "uploadDate": data['upload_date']
    }

    if data.get('duration'):
        schema["duration"] = data['duration']

    if data.get('content_url'):
        schema["contentUrl"] = data['content_url']

    if data.get('embed_url'):
        schema["embedUrl"] = data['embed_url']

    return schema


def format_json(schema):
    """Format schema as JSON-LD script tag."""
    json_str = json.dumps(schema, indent=2, ensure_ascii=False)
    return f'<script type="application/ld+json">\n{json_str}\n</script>'


# Schema type selection
schema_type = st.selectbox(
    "Select Schema Type",
    ["FAQ", "HowTo", "Article", "Product", "Local Business", "Breadcrumb", "Video"]
)

st.markdown("---")

# Dynamic form based on schema type
if schema_type == "FAQ":
    st.subheader("FAQ Schema Generator")
    st.markdown("Add question and answer pairs for your FAQ page.")

    num_faqs = st.number_input("Number of FAQs", min_value=1, max_value=20, value=3)

    faqs = []
    for i in range(int(num_faqs)):
        with st.expander(f"FAQ {i + 1}", expanded=i < 3):
            q = st.text_input(f"Question {i + 1}", key=f"q_{i}")
            a = st.text_area(f"Answer {i + 1}", key=f"a_{i}", height=100)
            faqs.append({"question": q, "answer": a})

    if st.button("Generate FAQ Schema", type="primary"):
        if any(faq['question'] and faq['answer'] for faq in faqs):
            schema = generate_faq_schema(faqs)
            st.code(format_json(schema), language="html")
            st.download_button("Download Schema", format_json(schema), "faq-schema.html", "text/html")
        else:
            st.warning("Please add at least one Q&A pair.")

elif schema_type == "HowTo":
    st.subheader("HowTo Schema Generator")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Title/Name*", placeholder="How to Change a Tire")
        total_time = st.text_input("Total Time (ISO 8601)", placeholder="PT30M", help="e.g., PT30M = 30 minutes")
    with col2:
        description = st.text_area("Description*", height=100)
        image = st.text_input("Image URL", placeholder="https://example.com/image.jpg")

    col1, col2 = st.columns(2)
    with col1:
        estimated_cost = st.text_input("Estimated Cost", placeholder="20")
    with col2:
        currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "CAD", "AUD"])

    st.markdown("### Steps")
    num_steps = st.number_input("Number of Steps", min_value=2, max_value=20, value=4)

    steps = []
    for i in range(int(num_steps)):
        with st.expander(f"Step {i + 1}", expanded=i < 4):
            step_name = st.text_input(f"Step {i + 1} Name", key=f"step_name_{i}", placeholder=f"Step {i + 1}")
            step_text = st.text_area(f"Step {i + 1} Instructions*", key=f"step_text_{i}")
            step_image = st.text_input(f"Step {i + 1} Image URL", key=f"step_img_{i}")
            steps.append({"name": step_name, "text": step_text, "image": step_image})

    if st.button("Generate HowTo Schema", type="primary"):
        if name and description and any(s['text'] for s in steps):
            data = {
                "name": name,
                "description": description,
                "total_time": total_time,
                "image": image,
                "estimated_cost": estimated_cost,
                "currency": currency,
                "steps": steps
            }
            schema = generate_howto_schema(data)
            st.code(format_json(schema), language="html")
            st.download_button("Download Schema", format_json(schema), "howto-schema.html", "text/html")
        else:
            st.warning("Please fill in required fields.")

elif schema_type == "Article":
    st.subheader("Article Schema Generator")

    article_type = st.selectbox("Article Type", ["Article", "NewsArticle", "BlogPosting"])

    col1, col2 = st.columns(2)
    with col1:
        headline = st.text_input("Headline*", placeholder="Article title")
        author_name = st.text_input("Author Name*")
        author_url = st.text_input("Author URL", placeholder="https://example.com/author")
        author_type = st.selectbox("Author Type", ["Person", "Organization"])
    with col2:
        description = st.text_area("Description*", height=100)
        date_published = st.date_input("Date Published*")
        date_modified = st.date_input("Date Modified (optional)")
        image = st.text_input("Image URL")

    col1, col2 = st.columns(2)
    with col1:
        publisher_name = st.text_input("Publisher Name*", placeholder="Your Site Name")
    with col2:
        publisher_logo = st.text_input("Publisher Logo URL")

    if st.button("Generate Article Schema", type="primary"):
        if headline and description and author_name and publisher_name:
            data = {
                "article_type": article_type,
                "headline": headline,
                "description": description,
                "author_name": author_name,
                "author_url": author_url,
                "author_type": author_type,
                "date_published": date_published.isoformat(),
                "date_modified": date_modified.isoformat() if date_modified != date_published else None,
                "publisher_name": publisher_name,
                "publisher_logo": publisher_logo,
                "image": image
            }
            schema = generate_article_schema(data)
            st.code(format_json(schema), language="html")
            st.download_button("Download Schema", format_json(schema), "article-schema.html", "text/html")
        else:
            st.warning("Please fill in required fields.")

elif schema_type == "Product":
    st.subheader("Product Schema Generator")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Product Name*")
        brand = st.text_input("Brand")
        sku = st.text_input("SKU")
        mpn = st.text_input("MPN (Manufacturer Part Number)")
        gtin = st.text_input("GTIN/UPC/EAN")
    with col2:
        description = st.text_area("Description*", height=150)
        image = st.text_input("Product Image URL")

    st.markdown("### Pricing")
    col1, col2, col3 = st.columns(3)
    with col1:
        price = st.text_input("Price*", placeholder="29.99")
        currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "CAD", "AUD"])
    with col2:
        availability = st.selectbox("Availability", ["InStock", "OutOfStock", "PreOrder", "BackOrder"])
        price_valid_until = st.date_input("Price Valid Until (optional)")
    with col3:
        url = st.text_input("Product URL")

    st.markdown("### Reviews (Optional)")
    col1, col2 = st.columns(2)
    with col1:
        rating_value = st.text_input("Average Rating", placeholder="4.5")
    with col2:
        review_count = st.text_input("Number of Reviews", placeholder="128")

    if st.button("Generate Product Schema", type="primary"):
        if name and description and price:
            data = {
                "name": name,
                "description": description,
                "brand": brand,
                "sku": sku,
                "mpn": mpn,
                "gtin": gtin,
                "image": image,
                "price": price,
                "currency": currency,
                "availability": availability,
                "price_valid_until": price_valid_until.isoformat() if price_valid_until else None,
                "url": url,
                "rating_value": rating_value,
                "review_count": review_count
            }
            schema = generate_product_schema(data)
            st.code(format_json(schema), language="html")
            st.download_button("Download Schema", format_json(schema), "product-schema.html", "text/html")
        else:
            st.warning("Please fill in required fields.")

elif schema_type == "Local Business":
    st.subheader("Local Business Schema Generator")

    business_types = [
        "LocalBusiness", "Restaurant", "HealthAndBeautyBusiness", "HomeAndConstructionBusiness",
        "LegalService", "Dentist", "MedicalClinic", "RealEstateAgent", "Store", "AutoRepair"
    ]
    business_type = st.selectbox("Business Type", business_types)

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Business Name*")
        phone = st.text_input("Phone Number")
        url = st.text_input("Website URL")
        description = st.text_area("Description")
    with col2:
        image = st.text_input("Image URL")
        price_range = st.selectbox("Price Range", ["", "$", "$$", "$$$", "$$$$"])

    st.markdown("### Address")
    col1, col2 = st.columns(2)
    with col1:
        street = st.text_input("Street Address*")
        city = st.text_input("City*")
        state = st.text_input("State/Region*")
    with col2:
        postal = st.text_input("Postal Code*")
        country = st.text_input("Country*", value="US")

    st.markdown("### Coordinates (Optional)")
    col1, col2 = st.columns(2)
    with col1:
        lat = st.text_input("Latitude")
    with col2:
        lng = st.text_input("Longitude")

    st.markdown("### Opening Hours (Optional)")
    add_hours = st.checkbox("Add opening hours")
    opening_hours = []
    if add_hours:
        days_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        num_specs = st.number_input("Number of hour specifications", 1, 7, 1)
        for i in range(int(num_specs)):
            col1, col2, col3 = st.columns(3)
            with col1:
                days = st.multiselect(f"Days {i+1}", days_options, key=f"days_{i}")
            with col2:
                opens = st.text_input(f"Opens {i+1}", placeholder="09:00", key=f"opens_{i}")
            with col3:
                closes = st.text_input(f"Closes {i+1}", placeholder="17:00", key=f"closes_{i}")
            opening_hours.append({"days": days, "open": opens, "close": closes})

    if st.button("Generate Local Business Schema", type="primary"):
        if name and street and city and state and postal and country:
            data = {
                "business_type": business_type,
                "name": name,
                "description": description,
                "phone": phone,
                "url": url,
                "image": image,
                "price_range": price_range,
                "street_address": street,
                "city": city,
                "state": state,
                "postal_code": postal,
                "country": country,
                "latitude": lat,
                "longitude": lng,
                "opening_hours": opening_hours
            }
            schema = generate_localbusiness_schema(data)
            st.code(format_json(schema), language="html")
            st.download_button("Download Schema", format_json(schema), "localbusiness-schema.html", "text/html")
        else:
            st.warning("Please fill in required fields.")

elif schema_type == "Breadcrumb":
    st.subheader("Breadcrumb Schema Generator")

    num_crumbs = st.number_input("Number of breadcrumbs", 2, 10, 3)

    breadcrumbs = []
    for i in range(int(num_crumbs)):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input(f"Breadcrumb {i+1} Name*", key=f"bc_name_{i}",
                                 placeholder="Home" if i == 0 else f"Category {i}")
        with col2:
            url = st.text_input(f"Breadcrumb {i+1} URL", key=f"bc_url_{i}",
                               placeholder="https://example.com" if i == 0 else "")
        breadcrumbs.append({"name": name, "url": url})

    if st.button("Generate Breadcrumb Schema", type="primary"):
        if any(bc['name'] for bc in breadcrumbs):
            schema = generate_breadcrumb_schema(breadcrumbs)
            st.code(format_json(schema), language="html")
            st.download_button("Download Schema", format_json(schema), "breadcrumb-schema.html", "text/html")
        else:
            st.warning("Please add at least one breadcrumb.")

elif schema_type == "Video":
    st.subheader("Video Schema Generator")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Video Title*")
        upload_date = st.date_input("Upload Date*")
        duration = st.text_input("Duration (ISO 8601)", placeholder="PT5M30S", help="e.g., PT5M30S = 5 min 30 sec")
    with col2:
        description = st.text_area("Description*", height=120)
        thumbnail = st.text_input("Thumbnail URL*")

    content_url = st.text_input("Content URL (direct video file)")
    embed_url = st.text_input("Embed URL (YouTube embed, etc.)")

    if st.button("Generate Video Schema", type="primary"):
        if name and description and thumbnail and upload_date:
            data = {
                "name": name,
                "description": description,
                "thumbnail_url": thumbnail,
                "upload_date": upload_date.isoformat(),
                "duration": duration,
                "content_url": content_url,
                "embed_url": embed_url
            }
            schema = generate_video_schema(data)
            st.code(format_json(schema), language="html")
            st.download_button("Download Schema", format_json(schema), "video-schema.html", "text/html")
        else:
            st.warning("Please fill in required fields.")

# Help section
with st.expander("About Schema Markup"):
    st.markdown("""
    **What is Schema Markup?**
    Schema markup (structured data) helps search engines understand your content better,
    potentially earning rich results in search (stars, FAQs, how-to steps, etc.).

    **How to Use:**
    1. Select your schema type
    2. Fill in the required fields
    3. Generate and copy the code
    4. Paste into your page's `<head>` section

    **Testing:**
    - [Google Rich Results Test](https://search.google.com/test/rich-results)
    - [Schema.org Validator](https://validator.schema.org/)
    """)

# Footer
st.markdown("---")
